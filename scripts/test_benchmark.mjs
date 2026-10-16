import { CompactNameClassifier } from './classifier.mjs';

async function runBenchmark() {
  console.log('===============================================================');
  console.log('Node.js Sub-Millisecond Classification Benchmark');
  console.log('===============================================================');

  const classifier = new CompactNameClassifier();
  await classifier.init();

  const testCases = [
    { text: 'Supratim Mondal', expected: 'REAL_NAME' },
    { text: 'Sayantika Biswas', expected: 'REAL_NAME' },
    { text: 'Ankush Dutta', expected: 'REAL_NAME' },
    { text: 'Priya Sharma', expected: 'REAL_NAME' },
    { text: 'John Smith', expected: 'REAL_NAME' },
    { text: 'Mar', expected: 'REAL_NAME' },
    { text: 'Kumar', expected: 'REAL_NAME' },
    { text: 'get', expected: 'NON_NAME' },
    { text: 'mar', expected: 'REAL_NAME' },
    { text: 'get mar', expected: 'NON_NAME' },
    { text: '0fxfps', expected: 'NON_NAME' },
    { text: 'ShadowHunter99', expected: 'NON_NAME' },
    { text: '112_SAYANTIKA BISWAS', expected: 'NON_NAME' },
    { text: 'AYAN CSE3048', expected: 'NON_NAME' },
    { text: 'toyexpo', expected: 'NON_NAME' },
    { text: 'fluffynsint', expected: 'NON_NAME' },
    { text: 'cecipfp', expected: 'NON_NAME' }
  ];

  // Warmup
  for (let i = 0; i < 20; i++) {
    await classifier.predict('Warmup Test');
  }

  console.log('\n--- Accuracy & Evaluation Results ---');
  let correct = 0;
  const latencies = [];

  for (const tc of testCases) {
    const t0 = performance.now();
    const res = await classifier.predict(tc.text);
    const dt = performance.now() - t0;
    latencies.push(dt);

    const match = res.classification === tc.expected;
    if (match) correct++;
    const icon = match ? '✅' : '❌';

    const p = res.raw_probs;
    const rawStr = `Real: ${(p.real_name * 100).toFixed(1)}% | Non: ${(p.non_name * 100).toFixed(1)}%`;
    console.log(
      `${icon} ${tc.text.padEnd(24)} => ${res.classification.padEnd(10)} (${(res.confidence * 100).toFixed(1)}%) | ${rawStr.padEnd(28)} [${dt.toFixed(3)} ms]`
    );
  }

  // 1000 iterations latency benchmark
  console.log('\n--- Latency Benchmark (1,000 Inferences on Single Thread) ---');
  const benchNames = ['Supratim Mondal', '0fxfps', 'Sayantika Biswas', 'ShadowHunter99'];
  const benchStart = performance.now();
  const N = 1000;
  for (let i = 0; i < N; i++) {
    await classifier.predict(benchNames[i % benchNames.length]);
  }
  const totalBenchTime = performance.now() - benchStart;
  const avgLatencyMs = totalBenchTime / N;

  console.log(`Total Time for ${N.toLocaleString()} inferences: ${totalBenchTime.toFixed(2)} ms`);
  console.log(`Average Latency per Name:              ${avgLatencyMs.toFixed(3)} ms`);
  console.log(`Throughput:                            ${Math.round(1000 / avgLatencyMs).toLocaleString()} queries/sec`);
  console.log(`Accuracy on Corner Cases:              ${correct}/${testCases.length} (${((correct / testCases.length) * 100).toFixed(1)}%)`);
  console.log('===============================================================\n');
}

runBenchmark().catch(console.error);
