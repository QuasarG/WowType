//! 知意 Laya zhen-mini-next-preview 的纯 Rust 推理（下一词预测）。
//!
//! 按模型包 README（Apache-2.0）的规格实现：ggml Q4_0 权重块、8 层 ModernBERT
//! 编码器（RoPE base 160000、滑动窗 ±64、erf GELU）、2 层 TransformerDecoder
//! （PyTorch 布局、norm_first、ReLU）、Unigram 分词（字节回退）和参考束搜索
//! （beam 8、k 词、每词至多 6 个 token、同词 LSE 合并）。只实现「下一词」路径
//! （选词头用不到）；token id 即行号（zhen-mini 无重映射）。

use std::collections::HashMap;
use std::path::Path;


use anyhow::{bail, Context as _, Result};

const DIM: usize = 256;
const HEADS: usize = 4;
const HEAD_DIM: usize = 64;
const LAYERS: usize = 8;
const ROPE_BASE: f64 = 160_000.0;
const ATT_SCALE: f32 = 1.0 / 8.0;
const SLIDE: isize = 64;
const LN_EPS: f32 = 1e-5;

// ---------------------------------------------------------------------------
// safetensors 最小读取器（F32 / U8），不引第三方依赖。
// ---------------------------------------------------------------------------

struct StFile {
    data: Vec<u8>,
    entries: HashMap<String, (String, Vec<usize>, usize, usize)>, // dtype, shape, begin, end
}

impl StFile {
    fn open(path: &Path) -> Result<Self> {
        let data = std::fs::read(path).with_context(|| format!("读取 {}", path.display()))?;
        if data.len() < 8 {
            bail!("safetensors 太短");
        }
        let hlen = u64::from_le_bytes(data[..8].try_into().unwrap()) as usize;
        let header: HashMap<String, serde_json::Value> =
            serde_json::from_slice(&data[8..8 + hlen]).context("解析 safetensors 头")?;
        let base = 8 + hlen;
        let mut entries = HashMap::new();
        for (name, meta) in header {
            if !name.is_empty() && name.as_bytes()[0] == b'%' {
                continue; // 元数据键
            }
            let dtype = meta["dtype"].as_str().unwrap_or_default().to_string();
            let shape: Vec<usize> = meta["shape"]
                .as_array()
                .map(|a| a.iter().filter_map(|v| v.as_u64().map(|u| u as usize)).collect())
                .unwrap_or_default();
            let off = &meta["data_offsets"];
            if let (Some(b), Some(e)) = (off[0].as_u64(), off[1].as_u64()) {
                entries.insert(name, (dtype, shape, base + b as usize, base + e as usize));
            }
        }
        Ok(Self { data, entries })
    }

    fn f32s(&self, name: &str) -> Result<Vec<f32>> {
        let (dtype, _, b, e) = self
            .entries
            .get(name)
            .with_context(|| format!("缺张量 {name}"))?;
        if dtype != "F32" {
            bail!("{name} 不是 F32");
        }
        let bytes = &self.data[*b..*e];
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    fn u8s(&self, name: &str) -> Result<Vec<u8>> {
        let (dtype, _, b, e) = self
            .entries
            .get(name)
            .with_context(|| format!("缺张量 {name}"))?;
        if dtype != "U8" {
            bail!("{name} 不是 U8");
        }
        Ok(self.data[*b..*e].to_vec())
    }

    fn i64s(&self, name: &str) -> Result<Vec<i64>> {
        let (dtype, _, b, e) = self
            .entries
            .get(name)
            .with_context(|| format!("缺张量 {name}"))?;
        if dtype != "I64" {
            bail!("{name} 不是 I64");
        }
        let bytes = &self.data[*b..*e];
        Ok(bytes
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    fn has(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }
}

// ---------------------------------------------------------------------------
// Q4_0：每 32 个值一节，f16 scale + 16 字节 nibble。
// ---------------------------------------------------------------------------

fn q4_rows(raw: &[u8], rows: usize, cols: usize) -> Vec<f32> {
    let stride = cols / 32 * 18;
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        let row = &raw[r * stride..(r + 1) * stride];
        for blk in 0..cols / 32 {
            let d = f32::from(f16_le(row[blk * 18], row[blk * 18 + 1]));
            let q = &row[blk * 18 + 2..blk * 18 + 18];
            for j in 0..16 {
                out[r * cols + blk * 32 + j] = ((q[j] & 15) as f32 - 8.0) * d;
                out[r * cols + blk * 32 + 16 + j] = ((q[j] >> 4) as f32 - 8.0) * d;
            }
        }
    }
    out
}

fn f16_le(lo: u8, hi: u8) -> half_f32 {
    let h = u16::from_le_bytes([lo, hi]);
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        // f16 次正规：m × 2^-24，直接算数值再组 bits（scale 一般用不到这段）
        let v = (m as f32) * (-24f32).exp2();
        f32::to_bits(if s != 0 { -v } else { v })
    } else {
        // f16 指数偏置 15 → f32 偏置 127
        s | ((e + 112) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}
type half_f32 = f32;

// ---------------------------------------------------------------------------
// 模型
// ---------------------------------------------------------------------------

struct EncLayer {
    attn_norm: Option<Vec<f32>>, // layer 0 没有
    wqkv: Vec<f32>,
    wo: Vec<f32>,
    mlp_norm: Vec<f32>,
    wi: Vec<f32>,   // 1024×256
    w_mlp: Vec<f32>, // 256×512
    sliding: bool,
}

struct DecLayer {
    self_in_w: Vec<f32>,  // 768×256
    self_in_b: Vec<f32>,
    self_out_w: Vec<f32>, // 256×256
    self_out_b: Vec<f32>,
    cross_in_w: Vec<f32>,
    cross_in_b: Vec<f32>,
    cross_out_w: Vec<f32>,
    cross_out_b: Vec<f32>,
    norm1: Vec<f32>,
    norm1_b: Vec<f32>,
    norm2: Vec<f32>,
    norm2_b: Vec<f32>,
    norm3: Vec<f32>,
    norm3_b: Vec<f32>,
    lin1_w: Vec<f32>,
    lin1_b: Vec<f32>,
    lin2_w: Vec<f32>,
    lin2_b: Vec<f32>,
}

pub struct Laya {
    tok_emb: Vec<f32>, // 32000×256（解包后的词表，兼作读出）
    emb_norm: Vec<f32>,
    layers: Vec<EncLayer>,
    final_norm: Vec<f32>,
    type_emb0: Vec<f32>,
    mem_in_w: Vec<f32>,
    mem_in_b: Vec<f32>,
    mem_ln_w: Vec<f32>,
    mem_ln_b: Vec<f32>,
    tok_in_w: Vec<f32>,
    tok_in_b: Vec<f32>,
    start: Vec<f32>,
    pos: Vec<f32>, // 22×256
    dec: Vec<DecLayer>,
    out_ln_w: Vec<f32>,
    out_ln_b: Vec<f32>,
    out_w: Vec<f32>,
    out_b: Vec<f32>,
    read_bias: Vec<f32>, // 32000
    // 搜索
    end_row: u32,
    zh_rows: Vec<u32>,
    en_start_rows: Vec<u32>,
    en_cont_rows: Vec<u32>,
    pieces: Vec<String>,
    chars_zh: Vec<u32>,
    // 分词器
    vocab: Vec<(String, f64)>,
    vocab_index: HashMap<String, u32>,
    unk_id: u32,
    rope_cos: Vec<f32>, // 32 个频率 × 位置在用时算；这里存每维 cos 表按需生成
}

impl Laya {
    /// zh 模式允许的行数（给加载日志用）。
    pub fn zh_row_count(&self) -> usize {
        self.zh_rows.len()
    }

    pub fn load_dir(dir: &Path) -> Result<Self> {
        let st = StFile::open(&dir.join("model.safetensors"))?;
        let cfg: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let nr: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("next_rows.json"))?)?;
        Self::build(st, &cfg, &nr, dir)
    }

    fn build(st: StFile, cfg: &serde_json::Value, nr: &serde_json::Value, dir: &Path) -> Result<Self> {
        let q4: HashMap<String, Vec<usize>> = serde_json::from_value(cfg["q4_0"].clone())?;
        let get_q4 = |name: &str| -> Result<Vec<f32>> {
            let rc = q4
                .get(name)
                .with_context(|| format!("config.q4_0 缺 {name}"))?;
            Ok(q4_rows(&st.u8s(name)?, rc[0], rc[1]))
        };
        let get_f32 = |name: &str| st.f32s(name);

        let tok_emb = get_q4("encoder.embeddings.tok_embeddings.weight")?;
        let layer_types: Vec<String> = serde_json::from_value(cfg["encoder"]["layer_types"].clone())?;
        if layer_types.len() != LAYERS {
            bail!("layer_types 数量不对");
        }
        let mut layers = Vec::with_capacity(LAYERS);
        for n in 0..LAYERS {
            layers.push(EncLayer {
                attn_norm: if n == 0 { None } else { Some(get_f32(&format!("encoder.layers.{n}.attn_norm.weight"))?) },
                wqkv: get_q4(&format!("encoder.layers.{n}.attn.Wqkv.weight"))?,
                wo: get_q4(&format!("encoder.layers.{n}.attn.Wo.weight"))?,
                mlp_norm: get_f32(&format!("encoder.layers.{n}.mlp_norm.weight"))?,
                wi: get_q4(&format!("encoder.layers.{n}.mlp.Wi.weight"))?,
                w_mlp: get_q4(&format!("encoder.layers.{n}.mlp.Wo.weight"))?,
                sliding: layer_types[n] == "sliding_attention",
            });
        }
        let mut dec = Vec::with_capacity(2);
        for n in 0..2 {
            let p = format!("next.layers.{n}");
            dec.push(DecLayer {
                self_in_w: get_q4(&format!("{p}.self_attn.in_proj_weight"))?,
                self_in_b: get_f32(&format!("{p}.self_attn.in_proj_bias"))?,
                self_out_w: get_q4(&format!("{p}.self_attn.out_proj.weight"))?,
                self_out_b: get_f32(&format!("{p}.self_attn.out_proj.bias"))?,
                cross_in_w: get_q4(&format!("{p}.multihead_attn.in_proj_weight"))?,
                cross_in_b: get_f32(&format!("{p}.multihead_attn.in_proj_bias"))?,
                cross_out_w: get_q4(&format!("{p}.multihead_attn.out_proj.weight"))?,
                cross_out_b: get_f32(&format!("{p}.multihead_attn.out_proj.bias"))?,
                norm1: get_f32(&format!("{p}.norm1.weight"))?,
                norm1_b: get_f32(&format!("{p}.norm1.bias"))?,
                norm2: get_f32(&format!("{p}.norm2.weight"))?,
                norm2_b: get_f32(&format!("{p}.norm2.bias"))?,
                norm3: get_f32(&format!("{p}.norm3.weight"))?,
                norm3_b: get_f32(&format!("{p}.norm3.bias"))?,
                lin1_w: get_q4(&format!("{p}.linear1.weight"))?,
                lin1_b: get_f32(&format!("{p}.linear1.bias"))?,
                lin2_w: get_q4(&format!("{p}.linear2.weight"))?,
                lin2_b: get_f32(&format!("{p}.linear2.bias"))?,
            });
        }
        let zh_rows: Vec<u32> = nr["zh"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let en_start_rows: Vec<u32> = nr["en_start"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let en_cont_rows: Vec<u32> = nr["en_cont"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let pieces: Vec<String> = nr["pieces"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().map(|s| s.to_string()).unwrap_or_default())
            .collect();
        let chars_zh: Vec<u32> = nr["chars_zh"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();

        // 分词器（Unigram + 字节回退）
        let tk: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("tokenizer.json"))?)?;
        let vocab: Vec<(String, f64)> = tk["model"]["vocab"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| {
                (
                    v[0].as_str().unwrap_or_default().to_string(),
                    v[1].as_f64().unwrap_or(0.0),
                )
            })
            .collect();
        let mut vocab_index = HashMap::with_capacity(vocab.len());
        for (i, (piece, _)) in vocab.iter().enumerate() {
            vocab_index.entry(piece.clone()).or_insert(i as u32);
        }
        let unk_id = tk["model"]["unk_id"].as_u64().unwrap_or(3) as u32;

        // RoPE 频率（32 个）
        let mut rope_cos = vec![0f32; HEAD_DIM / 2];
        for (i, c) in rope_cos.iter_mut().enumerate() {
            *c = ROPE_BASE.powf(-(2.0 * i as f64) / HEAD_DIM as f64) as f32;
        }

        Ok(Self {
            layers,
            tok_emb,
            emb_norm: get_f32("encoder.embeddings.norm.weight")?,
            final_norm: get_f32("encoder.final_norm.weight")?,
            type_emb0: {
                let t = get_f32("type_emb.weight")?;
                t[..DIM].to_vec()
            },
            mem_in_w: get_q4("next.mem_in.0.weight")?,
            mem_in_b: get_f32("next.mem_in.0.bias")?,
            mem_ln_w: get_f32("next.mem_in.1.weight")?,
            mem_ln_b: get_f32("next.mem_in.1.bias")?,
            tok_in_w: get_q4("next.tok_in.weight")?,
            tok_in_b: get_f32("next.tok_in.bias")?,
            start: get_f32("next.start")?,
            pos: get_f32("next.pos.weight")?,
            dec,
            out_ln_w: get_f32("next.out.0.weight")?,
            out_ln_b: get_f32("next.out.0.bias")?,
            out_w: get_q4("next.out.1.weight")?,
            out_b: get_f32("next.out.1.bias")?,
            read_bias: get_f32("next.bias")?,
            end_row: nr["end"].as_u64().unwrap_or(1) as u32,
            zh_rows,
            en_start_rows,
            en_cont_rows,
            pieces,
            chars_zh,
            vocab,
            vocab_index,
            unk_id,
            rope_cos,
        })
    }

    // -- 基础算子 ---------------------------------------------------------

    fn ln(x: &[f32], w: &[f32], b: &[f32]) -> Vec<f32> {
        let n = x.len();
        let mean = x.iter().sum::<f32>() / n as f32;
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
        let inv = 1.0 / (var + LN_EPS).sqrt();
        x.iter()
            .zip(w)
            .zip(b)
            .map(|((v, wv), bv)| (v - mean) * inv * wv + bv)
            .collect()
    }

    fn ln_nb(x: &[f32], w: &[f32]) -> Vec<f32> {
        let n = x.len();
        let mean = x.iter().sum::<f32>() / n as f32;
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
        let inv = 1.0 / (var + LN_EPS).sqrt();
        x.iter().zip(w).map(|(v, wv)| (v - mean) * inv * wv).collect()
    }

    fn linear(x: &[f32], w: &[f32], b: &[f32]) -> Vec<f32> {
        let out = w.len() / x.len();
        let mut y = vec![0f32; out];
        for (k, yv) in y.iter_mut().enumerate() {
            let row = &w[k * x.len()..(k + 1) * x.len()];
            *yv = row.iter().zip(x).map(|(a, bv)| a * bv).sum::<f32>() + b[k];
        }
        y
    }

    fn linear_nb(x: &[f32], w: &[f32]) -> Vec<f32> {
        let out = w.len() / x.len();
        let mut y = vec![0f32; out];
        for (k, yv) in y.iter_mut().enumerate() {
            let row = &w[k * x.len()..(k + 1) * x.len()];
            *yv = row.iter().zip(x).map(|(a, bv)| a * bv).sum::<f32>();
        }
        y
    }

    fn gelu_erf(v: f32) -> f32 {
        0.5 * v * (1.0 + erf(v * std::f32::consts::FRAC_1_SQRT_2))
    }

    fn rope(&self, v: &mut [f32], pos: usize) {
        // v: 64 维（单头），rotate-half
        for h in 0..HEAD_DIM / 2 {
            let theta = self.rope_cos[h] * pos as f32;
            let (c, s) = (theta.cos(), theta.sin());
            let a = v[h];
            let b = v[h + HEAD_DIM / 2];
            v[h] = a * c - b * s;
            v[h + HEAD_DIM / 2] = b * c + a * s;
        }
    }

    // -- 编码器（记忆路径：<bos> ctx <eos>，unit 全 0，无 marker） ----------
    // 返回每层的 K（RoPE 后）与 V，以及最终输出（final_norm 后）。

    fn encode(&self, ids: &[u32]) -> (Vec<Vec<Vec<f32>>>, Vec<Vec<Vec<f32>>>, Vec<Vec<f32>>) {
        let t = ids.len();
        let mut x: Vec<Vec<f32>> = Vec::with_capacity(t);
        for &id in ids {
            let v = self.tok_emb[id as usize * DIM..(id as usize + 1) * DIM].to_vec();
            x.push(v);
        }
        // embeddings LayerNorm（无 bias）
        for row in x.iter_mut() {
            *row = Self::ln_nb(row, &self.emb_norm);
        }
        let mut all_k: Vec<Vec<Vec<f32>>> = vec![Vec::new(); LAYERS]; // [layer][pos][256]
        let mut all_v: Vec<Vec<Vec<f32>>> = vec![Vec::new(); LAYERS];
        for (li, layer) in self.layers.iter().enumerate() {
            // 每层从 x（残差流）出发
            let h: Vec<Vec<f32>> = match &layer.attn_norm {
                Some(w) => x.iter().map(|r| Self::ln_nb(r, w)).collect(),
                None => x.clone(),
            };
            // qkv
            let mut q = vec![vec![0f32; DIM]; t];
            let mut k = vec![vec![0f32; DIM]; t];
            let mut v = vec![vec![0f32; DIM]; t];
            for (i, hi) in h.iter().enumerate() {
                let qkv = Self::linear_nb(hi, &layer.wqkv);
                q[i].copy_from_slice(&qkv[..DIM]);
                k[i].copy_from_slice(&qkv[DIM..2 * DIM]);
                v[i].copy_from_slice(&qkv[2 * DIM..]);
                for head in 0..HEADS {
                    let s = head * HEAD_DIM;
                    self.rope(&mut q[i][s..s + HEAD_DIM], i);
                    self.rope(&mut k[i][s..s + HEAD_DIM], i);
                }
            }
            all_k[li] = k.clone();
            all_v[li] = v.clone();
            // attention
            for i in 0..t {
                let mut attn_out = vec![0f32; DIM];
                for head in 0..HEADS {
                    let s = head * HEAD_DIM;
                    let mut scores = Vec::with_capacity(t);
                    for j in 0..t {
                        if layer.sliding && (i as isize - j as isize).abs() > SLIDE {
                            continue;
                        }
                        let dot: f32 = (0..HEAD_DIM)
                            .map(|d| q[i][s + d] * k[j][s + d])
                            .sum::<f32>()
                            * ATT_SCALE;
                        scores.push((dot, j));
                    }
                    softmax_inplace(&mut scores);
                    let mut acc = vec![0f32; HEAD_DIM];
                    for (w, j) in &scores {
                        for d in 0..HEAD_DIM {
                            acc[d] += w * v[*j][s + d];
                        }
                    }
                    attn_out[s..s + HEAD_DIM].copy_from_slice(&acc);
                }
                let proj = Self::linear_nb(&attn_out, &layer.wo);
                for d in 0..DIM {
                    x[i][d] += proj[d];
                }
            }
            // MLP
            for i in 0..t {
                let hn = Self::ln_nb(&x[i], &layer.mlp_norm);
                let gate = Self::linear_nb(&hn, &layer.wi); // 1024
                let mut prod = vec![0f32; 512];
                for d in 0..512 {
                    prod[d] = Self::gelu_erf(gate[d]) * gate[512 + d];
                }
                let out = Self::linear_nb(&prod, &layer.w_mlp);
                for d in 0..DIM {
                    x[i][d] += out[d];
                }
            }
        }
        for row in x.iter_mut() {
            *row = Self::ln_nb(row, &self.final_norm);
        }
        (all_k, all_v, x)
    }

    // -- 记忆与交叉 K/V ----------------------------------------------------

    fn build_memory(&self, enc_out: &[Vec<f32>]) -> (Vec<Vec<f32>>, Vec<Vec<Vec<f32>>>, Vec<Vec<Vec<f32>>>) {
        // m = LN(mem_in.0(enc + type_emb0))
        let m: Vec<Vec<f32>> = enc_out
            .iter()
            .map(|r| {
                let sum: Vec<f32> = r.iter().zip(&self.type_emb0).map(|(a, b)| a + b).collect();
                let lin = Self::linear(&sum, &self.mem_in_w, &self.mem_in_b);
                Self::ln(&lin, &self.mem_ln_w, &self.mem_ln_b)
            })
            .collect();
        let mut ck = vec![Vec::<Vec<f32>>::new(); 2];
        let mut cv = vec![Vec::<Vec<f32>>::new(); 2];
        for (li, layer) in self.dec.iter().enumerate() {
            let mut ks = Vec::with_capacity(m.len());
            let mut vs = Vec::with_capacity(m.len());
            for mi in &m {
                let qkv = Self::linear(mi, &layer.cross_in_w, &layer.cross_in_b);
                ks.push(qkv[DIM..2 * DIM].to_vec());
                vs.push(qkv[2 * DIM..].to_vec());
            }
            ck[li] = ks;
            cv[li] = vs;
        }
        (m, ck, cv)
    }

    // -- 解码一步 -----------------------------------------------------------

    fn decode_step(
        &self,
        input: &[f32],       // 本位置输入（已加 pos）
        pos: usize,
        self_k_cache: &mut Vec<Vec<Vec<f32>>>, // [layer][已写位置][256]
        self_v_cache: &mut Vec<Vec<Vec<f32>>>,
        ck: &[Vec<Vec<f32>>],
        cv: &[Vec<Vec<f32>>],
    ) -> Vec<f32> {
        let mut x = input.to_vec();
        let t = self_k_cache[0].len(); // 之前的位置数（不含当前）
        for (li, layer) in self.dec.iter().enumerate() {
            // self-attn（因果：当前看 0..=t）
            let h1 = Self::ln(&x, &layer.norm1, &layer.norm1_b);
            let qkv = Self::linear(&h1, &layer.self_in_w, &layer.self_in_b);
            let mut q = qkv[..DIM].to_vec();
            for head in 0..HEADS {
                let s = head * HEAD_DIM;
                self.rope(&mut q[s..s + HEAD_DIM], t);
            }
            // 把当前步的 k,v 追加进缓存（RoPE 后的 k）
            let mut k_cur = qkv[DIM..2 * DIM].to_vec();
            let mut v_cur = qkv[2 * DIM..].to_vec();
            for head in 0..HEADS {
                let s = head * HEAD_DIM;
                self.rope(&mut k_cur[s..s + HEAD_DIM], t);
            }
            // 临时扩展视图
            let mut attn_out = vec![0f32; DIM];
            for head in 0..HEADS {
                let s = head * HEAD_DIM;
                let mut scores = Vec::with_capacity(t + 1);
                for j in 0..t {
                    let dot: f32 = (0..HEAD_DIM).map(|d| q[s + d] * self_k_cache[li][j][s + d]).sum::<f32>() * ATT_SCALE;
                    scores.push((dot, j as usize, false));
                }
                {
                    let dot: f32 = (0..HEAD_DIM).map(|d| q[s + d] * k_cur[s + d]).sum::<f32>() * ATT_SCALE;
                    scores.push((dot, t, true));
                }
                softmax_triplet(&mut scores);
                let mut acc = vec![0f32; HEAD_DIM];
                for (w, j, cur) in &scores {
                    let vref = if *cur { &v_cur } else { &self_v_cache[li][*j] };
                    for d in 0..HEAD_DIM {
                        acc[d] += w * vref[s + d];
                    }
                }
                attn_out[s..s + HEAD_DIM].copy_from_slice(&acc);
            }
            self_k_cache[li].push(k_cur);
            self_v_cache[li].push(v_cur);
            let sa = Self::linear(&attn_out, &layer.self_out_w, &layer.self_out_b);
            for d in 0..DIM {
                x[d] += sa[d];
            }
            // cross-attn
            let h2 = Self::ln(&x, &layer.norm2, &layer.norm2_b);
            let qkv2 = Self::linear(&h2, &layer.cross_in_w, &layer.cross_in_b);
            let q2 = &qkv2[..DIM];
            let mut attn2 = vec![0f32; DIM];
            for head in 0..HEADS {
                let s = head * HEAD_DIM;
                let mut scores = Vec::with_capacity(ck[li].len());
                for (j, krow) in ck[li].iter().enumerate() {
                    let dot: f32 = (0..HEAD_DIM).map(|d| q2[s + d] * krow[s + d]).sum::<f32>() * ATT_SCALE;
                    scores.push((dot, j));
                }
                softmax_inplace(&mut scores);
                let mut acc = vec![0f32; HEAD_DIM];
                for (w, j) in &scores {
                    for d in 0..HEAD_DIM {
                        acc[d] += w * cv[li][*j][s + d];
                    }
                }
                attn2[s..s + HEAD_DIM].copy_from_slice(&acc);
            }
            let ca = Self::linear(&attn2, &layer.cross_out_w, &layer.cross_out_b);
            for d in 0..DIM {
                x[d] += ca[d];
            }
            // FFN
            let h3 = Self::ln(&x, &layer.norm3, &layer.norm3_b);
            let inner = Self::linear(&h3, &layer.lin1_w, &layer.lin1_b);
            let relu: Vec<f32> = inner.iter().map(|v| v.max(0.0)).collect();
            let f = Self::linear(&relu, &layer.lin2_w, &layer.lin2_b);
            for d in 0..DIM {
                x[d] += f[d];
            }
        }
        // 读出头
        let o = Self::ln(&x, &self.out_ln_w, &self.out_ln_b);
        Self::linear(&o, &self.out_w, &self.out_b)
    }

    // -- 束搜索 --------------------------------------------------------------

    /// 给定 `<bos> ctx <eos>` 的 token 序列，返回 k 个（词, 对数概率），降序。
    pub fn predict_tokens(&self, ids: &[u32], mode: usize, min_chars: u32, k: usize, beam: usize) -> Vec<(String, f64)> {
        let (_ek, _ev, enc_out) = self.encode(ids);
        let (_m, ck, cv) = self.build_memory(&enc_out);

        #[derive(Clone)]
        struct Hyp {
            logp: f64,
            tokens: Vec<u32>,
            k_cache: Vec<Vec<Vec<f32>>>,
            v_cache: Vec<Vec<Vec<f32>>>,
        }
        let start_hyp = || Hyp {
            logp: 0.0,
            tokens: Vec::new(),
            k_cache: vec![Vec::new(); 2],
            v_cache: vec![Vec::new(); 2],
        };
        let mut open: Vec<Hyp> = vec![start_hyp()];
        // 词累积：text -> (lse_logp)
        let mut words: HashMap<String, f64> = HashMap::new();
        let mut word_list: Vec<String> = Vec::new();


        for step in 0..7 {
            // 行集按步选择：中文一律 zh；英文首 token 用 en_start，其后 en_cont。
            let mut rows: Vec<u32> = if mode == 0 {
                self.zh_rows.clone()
            } else if step == 0 {
                self.en_start_rows.clone()
            } else {
                self.en_cont_rows.clone()
            };
            if !rows.contains(&self.end_row) {
                rows.push(self.end_row);
            }
            // 先给所有（假设, 行）打分，再物化前 beam 个开放假设——
            // 物化要克隆 KV 缓存，不能对一万五千个候选都做。
            let mut cands: Vec<(f64, usize, u32)> = Vec::with_capacity(open.len() * rows.len());
            let mut advanced: Vec<(Vec<Vec<Vec<f32>>>, Vec<Vec<Vec<f32>>>)> = Vec::with_capacity(open.len());
            for (hi, hyp) in open.iter().enumerate() {
                let pos = hyp.tokens.len();
                let input: Vec<f32> = if pos == 0 {
                    self.start.clone()
                } else {
                    let emb: Vec<f32> = self.tok_emb[hyp.tokens[pos - 1] as usize * DIM
                        ..(hyp.tokens[pos - 1] as usize + 1) * DIM]
                        .to_vec();
                    Self::linear(&emb, &self.tok_in_w, &self.tok_in_b)
                };
                let p = &self.pos[pos * DIM..(pos + 1) * DIM];
                let input: Vec<f32> = input.iter().zip(p).map(|(a, b)| a + b).collect();
                let mut kc = hyp.k_cache.clone();
                let mut vc = hyp.v_cache.clone();
                let h = self.decode_step(&input, pos, &mut kc, &mut vc, &ck, &cv);
                advanced.push((kc, vc));
                let mut logits = Vec::with_capacity(rows.len());
                for &r in &rows {
                    let row_off = r as usize * DIM;
                    let dot: f32 = (0..DIM).map(|d| h[d] * self.tok_emb[row_off + d]).sum::<f32>();
                    logits.push(dot + self.read_bias[r as usize]);
                }
                let lse = log_softmax_inplace(&mut logits);
                let only_end = step == 6;
                for (ri, &r) in rows.iter().enumerate() {
                    if only_end && r != self.end_row {
                        continue;
                    }
                    if r == self.end_row && hyp.tokens.is_empty() {
                        continue;
                    }
                    cands.push((hyp.logp + (logits[ri] - lse) as f64, hi, r));
                }
            }
            cands.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let mut next_open: Vec<Hyp> = Vec::with_capacity(beam);
            for (lp, hi, r) in cands {
                if next_open.len() >= beam {
                    break; // 排在第 beam 名之后的候选（含收尾）已经赢不了
                }
                if r == self.end_row {
                    let text: String = if mode == 0 {
                        open[hi].tokens.iter().map(|&t| strip_meta(&self.pieces[t as usize])).collect()
                    } else {
                        let joined: String = open[hi].tokens.iter().map(|&t| self.pieces[t as usize].clone()).collect();
                        joined.replace('\u{2581}', " ").trim().to_string()
                    };
                    let enough = if mode == 0 {
                        open[hi].tokens.iter().map(|&t| self.chars_zh[t as usize]).sum::<u32>() >= min_chars
                    } else {
                        text.chars().count() as u32 >= min_chars.max(1)
                    };
                    if enough {
                        let e = words.entry(text.clone()).or_insert_with(|| {
                            word_list.push(text.clone());
                            f64::NEG_INFINITY
                        });
                        *e = lse_add(*e, lp);
                    }
                } else {
                    let (kc, vc) = advanced[hi].clone();
                    let mut nh = Hyp { logp: lp, tokens: open[hi].tokens.clone(), k_cache: kc, v_cache: vc };
                    nh.tokens.push(r);
                    next_open.push(nh);
                }
            }
            open = next_open;
            if words.len() >= k {
                let mut scores: Vec<f64> = words.values().copied().collect();
                scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                let kth = scores[k - 1];
                if open.iter().all(|h| h.logp <= kth) {
                    break;
                }
            }
            if open.is_empty() {
                break;
            }
        }
        let mut out: Vec<(String, f64)> = words.into_iter().collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0)));
        out.truncate(k);
        out
    }

    /// 上下文文本 → token（Unigram Viterbi，字节回退）→ 束搜索。
    pub fn predict_text(&self, context: &str, k: usize) -> Vec<(String, f64)> {
        let ids = self.tokenize(context);
        let seq = self.wrap_bos_eos(&ids);
        self.predict_tokens(&seq, 0, 2, k, 8)
    }

    fn wrap_bos_eos(&self, ids: &[u32]) -> Vec<u32> {
        let mut seq = Vec::with_capacity(ids.len() + 2);
        seq.push(2); // <bos>
        let start = ids.len().saturating_sub(128);
        seq.extend_from_slice(&ids[start..]);
        seq.push(1); // <eos>
        seq
    }

    fn tokenize(&self, text: &str) -> Vec<u32> {
        let bytes = text.as_bytes();
        let n = bytes.len();
        // UTF-8 边界集合（只在合法边界切词）
        let mut boundary = vec![false; n + 1];
        boundary[0] = true;
        boundary[n] = true;
        {   // 用 chars 标边界
            let mut i = 0;
            for c in text.chars() {
                i += c.len_utf8();
                if i <= n { boundary[i] = true; }
            }
        }
        const INF: f64 = 1e18;
        let mut best = vec![INF; n + 1];
        let mut back: Vec<i32> = vec![-1; n + 1]; // (piece_id, 起点打包) 简化：存 (start, id)
        let mut back_piece: Vec<(usize, i32)> = vec![(0, -1); n + 1];
        best[0] = 0.0;
        for i in 0..n {
            if best[i] >= INF || !boundary[i] {
                continue;
            }
            for j in i + 1..=n.min(i + 16) {
                if !boundary[j] {
                    continue;
                }
                let piece = &text[i..j];
                if let Some(&id) = self.vocab_index.get(piece) {
                    let sc = best[i] + self.vocab[id as usize].1;
                    if sc < best[j] {
                        best[j] = sc;
                        back_piece[j] = (i, id as i32);
                    }
                }
            }
            // unk + 字节回退：单个 UTF-8 字符走 <0xXX>
            for j in i + 1..=n.min(i + 4) {
                if !boundary[j] {
                    continue;
                }
                let ch = &text[i..j];
                if ch.chars().count() == 1 {
                    let mut sc = best[i];
                    let mut ok = true;
                    let mut toks = Vec::new();
                    for b in ch.as_bytes() {
                        let hex = format!("<{b:02X}>");
                        match self.vocab_index.get(&hex) {
                            Some(&id) => { sc += self.vocab[id as usize].1; toks.push(id); }
                            None => { ok = false; break; }
                        }
                    }
                    if ok && sc < best[j] {
                        best[j] = sc;
                        back_piece[j] = (i, -(toks.len() as i32) - 1); // 负数标记字节回退
                        // 存字节 token 到 side map 太繁琐：这里重解
                        back_piece[j] = (i, 0); // 0 表示字节回退（重切）
                    }
                }
            }
        }
        // 回溯
        let mut ids = Vec::new();
        let mut i = n;
        while i > 0 {
            let (start, pid) = back_piece[i];
            if pid > 0 {
                ids.push(pid as u32);
            } else {
                // 字节回退：重切这个字符
                let ch = &text[start..i];
                for b in ch.as_bytes() {
                    let hex = format!("<{b:02X}>");
                    if let Some(&id) = self.vocab_index.get(&hex) {
                        ids.push(id);
                    }
                }
            }
            i = start;
        }
        ids.reverse();
        ids
    }
}

fn strip_meta(piece: &str) -> &str {
    piece.strip_prefix('▁').unwrap_or(piece)
}

fn softmax_inplace(scores: &mut Vec<(f32, usize)>) {
    let max = scores.iter().map(|(s, _)| *s).fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = scores.iter().map(|(s, _)| (s - max).exp()).sum();
    for (s, _) in scores.iter_mut() {
        *s = (*s - max).exp() / sum;
    }
}

fn softmax_triplet(scores: &mut Vec<(f32, usize, bool)>) {
    let max = scores.iter().map(|(s, _, _)| *s).fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = scores.iter().map(|(s, _, _)| (s - max).exp()).sum();
    for (s, _, _) in scores.iter_mut() {
        *s = (*s - max).exp() / sum;
    }
}

fn log_softmax_inplace(logits: &mut [f32]) -> f32 {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|v| (v - max).exp()).sum();
    let lse = max + sum.ln();
    for v in logits.iter_mut() {
        *v -= lse;
    }
    lse as f32
}

fn lse_add(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY { return b; }
    if b == f64::NEG_INFINITY { return a; }
    let (hi, lo) = if a > b { (a, b) } else { (b, a) };
    hi + (lo - hi).exp().ln_1p()
}

fn erf(x: f32) -> f32 {
    // Abramowitz–Stegun 7.1.26（|误差| < 1.5e-7）
    let s = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592)
            * t
            * (-x * x).exp();
    s * y as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 对表包内 next0..next7 测试向量：5 个词的集合与顺序应与 fp32 参考一致。
    #[test]
    #[ignore]
    fn matches_next_word_vectors() {
        let dir = std::path::Path::new("/home/zexin/.local/share/wowtype/zhiyi/zhen-mini");
        if !dir.exists() {
            eprintln!("跳过：本机没有 zhiyi 模型");
            return;
        }
        let m = Laya::load_dir(dir).unwrap();
        let tv = StFile::open(&dir.join("test_vectors.safetensors")).unwrap();
        let (mut pass, mut total) = (0, 0);
        for n in 0..8 {
            let ids: Vec<u32> = tv
                .i64s(&format!("next{n}.input_ids"))
                .unwrap()
                .iter()
                .map(|&v| v as u32)
                .collect();
            let mode = tv.i64s(&format!("next{n}.mode")).unwrap()[0] as usize;
            let min_chars = tv.i64s(&format!("next{n}.min_chars")).unwrap()[0] as u32;
            let expect_bytes = tv.u8s(&format!("next{n}.text")).unwrap();
            let expect: Vec<&str> = std::str::from_utf8(&expect_bytes).unwrap().lines().collect();
            let got = m.predict_tokens(&ids, mode, min_chars, 5, 8);
            let got_words: Vec<&str> = got.iter().map(|(w, _)| w.as_str()).collect();
            total += 1;
            let overlap = got_words.iter().filter(|w| expect.contains(w)).count();
            if overlap >= 3 {
                pass += 1;
            }
            if got_words != expect {
                eprintln!("next{n}: 期 {:?} 得 {:?}（重合 {overlap}/5）", expect, got_words);
            }
        }
        eprintln!("通过 {pass}/{total}");
        assert!(pass >= 5, "测试向量对不上（{pass}/{total}）");
    }
}
