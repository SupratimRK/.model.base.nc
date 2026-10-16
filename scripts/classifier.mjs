import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import ort from 'onnxruntime-node';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT_DIR = path.resolve(__dirname, '..');

const MODEL_PATH = path.join(ROOT_DIR, 'models', 'model_int8.onnx');
const VOCAB_PATH = path.join(ROOT_DIR, 'models', 'tokenizer', 'vocab.txt');
const CONFIG_PATH = path.join(ROOT_DIR, 'config', 'model_config.json');

export class CompactNameClassifier {
  constructor() {
    this.session = null;
    this.vocab = new Map();
    this.idToVocab = new Map();
    this.maxLength = 20;
    this.isReady = false;

    // Special token IDs
    this.padId = 0;
    this.unkId = 1;
    this.clsId = 2;
    this.sepId = 3;
    this.maskId = 4;
  }

  async init(modelPath = MODEL_PATH) {
    if (this.isReady) return;

    // 1. Load Vocab
    if (!fs.existsSync(VOCAB_PATH)) {
      throw new Error(`Vocabulary file not found at: ${VOCAB_PATH}`);
    }
    const lines = fs.readFileSync(VOCAB_PATH, 'utf-8').split(/\r?\n/);
    for (let i = 0; i < lines.length; i++) {
      const token = lines[i];
      if (token.length > 0) {
        this.vocab.set(token, i);
        this.idToVocab.set(i, token);
      }
    }

    // 2. Load Config if available
    if (fs.existsSync(CONFIG_PATH)) {
      try {
        const cfg = JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf-8'));
        this.maxLength = cfg.max_length || 20;
      } catch (e) {}
    }

    // 3. Initialize ONNX Session with single-thread optimization
    const sessionOptions = {
      intraOpNumThreads: 1,
      interOpNumThreads: 1,
      executionMode: 'sequential',
      graphOptimizationLevel: 'all'
    };

    this.session = await ort.InferenceSession.create(modelPath, sessionOptions);
    this.isReady = true;
  }

  tokenize(text) {
    if (!text || typeof text !== 'string') return [this.clsId, this.sepId];

    // Pre-tokenize: Whitespace, Punctuation, and Individual Digits
    // Split text into tokens separating punctuation and digits
    const rawTokens = text.match(/[\p{L}]+|\d|[^\s\p{L}\d]/gu) || [];
    const tokenIds = [this.clsId];

    for (const chunk of rawTokens) {
      if (tokenIds.length >= this.maxLength - 1) break;

      // If chunk is a single digit or in vocab directly
      if (this.vocab.has(chunk)) {
        tokenIds.push(this.vocab.get(chunk));
        continue;
      }

      // WordPiece greedy subword matching
      let start = 0;
      let isBad = false;
      const subTokens = [];

      while (start < chunk.length) {
        let end = chunk.length;
        let curSubId = null;

        while (start < end) {
          const sub = (start === 0) ? chunk.slice(start, end) : '##' + chunk.slice(start, end);
          if (this.vocab.has(sub)) {
            curSubId = this.vocab.get(sub);
            break;
          }
          end--;
        }

        if (curSubId === null) {
          isBad = true;
          break;
        }

        subTokens.push(curSubId);
        start = end;
      }

      if (isBad) {
        tokenIds.push(this.unkId);
      } else {
        for (const tid of subTokens) {
          if (tokenIds.length >= this.maxLength - 1) break;
          tokenIds.push(tid);
        }
      }
    }

    tokenIds.push(this.sepId);

    // Padding up to maxLength
    const inputIds = new Int32Array(this.maxLength);
    const attentionMask = new Int32Array(this.maxLength);

    for (let i = 0; i < this.maxLength; i++) {
      if (i < tokenIds.length) {
        inputIds[i] = tokenIds[i];
        attentionMask[i] = 1;
      } else {
        inputIds[i] = this.padId;
        attentionMask[i] = 0;
      }
    }

    return { inputIds, attentionMask };
  }

  softmax(arr) {
    const maxVal = Math.max(arr[0], arr[1]);
    const exp0 = Math.exp(arr[0] - maxVal);
    const exp1 = Math.exp(arr[1] - maxVal);
    const sum = exp0 + exp1;
    return [exp0 / sum, exp1 / sum];
  }

  async predict(text) {
    if (!this.isReady) await this.init();

    const { inputIds, attentionMask } = this.tokenize(text);

    // Construct Int32 Tensors
    const tensorIds = new ort.Tensor('int32', inputIds, [1, this.maxLength]);
    const tensorMask = new ort.Tensor('int32', attentionMask, [1, this.maxLength]);

    const feeds = {
      input_ids: tensorIds,
      attention_mask: tensorMask
    };

    const outputMap = await this.session.run(feeds);
    const logits = outputMap.logits.data; // Float32Array [non_name_logit, real_name_logit]
    const probs = this.softmax(logits);

    const isReal = probs[1] >= probs[0];
    const classification = isReal ? 'REAL_NAME' : 'NON_NAME';
    const confidence = isReal ? probs[1] : probs[0];

    return {
      text,
      classification,
      confidence,
      raw_probs: {
        non_name: probs[0],
        real_name: probs[1]
      }
    };
  }
}

// Standalone CLI execution
if (process.argv[1] && process.argv[1].endsWith('classifier.mjs')) {
  const query = process.argv.slice(2).join(' ') || 'Supratim Mondal';
  const classifier = new CompactNameClassifier();
  await classifier.init();
  const t0 = performance.now();
  const res = await classifier.predict(query);
  const dt = performance.now() - t0;
  console.log(`\nInput:          "${res.text}"`);
  console.log(`Classification: ${res.classification} (${(res.confidence * 100).toFixed(1)}%)`);
  console.log(`Raw:            Real: ${(res.raw_probs.real_name * 100).toFixed(2)}% | Non: ${(res.raw_probs.non_name * 100).toFixed(2)}%`);
  console.log(`Latency:        ${dt.toFixed(3)} ms\n`);
}
