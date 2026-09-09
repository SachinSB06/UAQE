# GitHub CI Fix Report: Frontend TypeScript Test Runner

**Status**: Resolved & Verified  
**Date**: 2026-09-09  
**Repository**: [SachinSB06/UAQE](https://github.com/SachinSB06/UAQE)  

---

## 1. Original Frontend CI Error

The GitHub Actions workflow failed at the job step **Frontend Verification & Production Build** during:
```bash
node src/tests/test_candidate_metrics.mjs
```

### Error Stack Trace
```
TypeError [ERR_UNKNOWN_FILE_EXTENSION]: Unknown file extension ".ts" for /home/runner/work/UAQE/UAQE/frontend/src/utils/candidateMetrics.ts
    at new NodeError (node:internal/errors:405:5)
    at Object.getFileProtocolModuleFormat [as file:] (node:internal/modules/esm/get_format:79:11)
    at defaultGetFormat (node:internal/modules/esm/get_format:121:38)
    at defaultLoad (node:internal/modules/esm/load:81:20)
    at nextLoad (node:internal/modules/esm/loader:163:28)
    at ESMLoader.load (node:internal/modules/esm/loader:605:26)
    at ESMLoader.moduleProvider (node:internal/modules/esm/loader:457:22)
    at new ModuleJob (node:internal/modules/esm/module_job:64:26)
    at ESMLoader.getModuleJob (node:internal/modules/esm/loader:421:17)
    at async ModuleWrap.<anonymous> (node:internal/modules/esm/module_job:79:21) {
  code: 'ERR_UNKNOWN_FILE_EXTENSION'
}
```

---

## 2. Root Cause

1. `frontend/src/tests/test_candidate_metrics.mjs` directly imports production TypeScript utilities:
   ```javascript
   import { getActiveCandidateMetrics } from '../utils/candidateMetrics.ts';
   import { normalizeCandidate } from '../utils/candidateNormalizer.ts';
   ```
2. In GitHub Actions, the runner environment uses **Node.js 20** (`actions/setup-node@v4` with `node-version: 20`). Plain Node 20 runtime does not natively understand or strip TypeScript syntax/extensions (`.ts`) when executing ECMAScript modules (`.mjs`).
3. While newer Node versions (e.g. Node 22.6+ / 24) feature experimental type-stripping flags, Node 20 requires a TypeScript loader/transpiler (such as `tsx`) to execute TypeScript-dependent ESM test scripts.

---

## 3. Exact Fix

1. **Installed `tsx` as Dev Dependency**:
   Added `tsx` (`^4.23.13`) to `frontend/package.json` devDependencies and locked in `frontend/package-lock.json`. `tsx` provides zero-config TypeScript execution powered by `esbuild`, natively resolving `.ts` extensions and type annotations without modifying the production code.
2. **Added `npm test` Script**:
   Configured in `frontend/package.json`:
   ```json
   "test": "tsx src/tests/test_candidate_metrics.mjs && tsx src/tests/test_candidate_normalization.mjs"
   ```
3. **Updated GitHub Actions Workflow**:
   In `.github/workflows/tests.yml`, updated the test execution step from `node` to `npx tsx`:
   ```yaml
         - name: Run Candidate Metrics Logic Unit Tests
           working-directory: ./frontend
           run: |
             npx tsx src/tests/test_candidate_metrics.mjs
             npx tsx src/tests/test_candidate_normalization.mjs
   ```
4. **Preserved Production Source & Behavior**:
   - Zero changes to `frontend/src/utils/candidateMetrics.ts`
   - Zero changes to `frontend/src/utils/candidateNormalizer.ts`
   - Zero changes to `frontend/src/pages/ResultsPage.tsx`
   - Zero test assertions mocked, bypassed, or weakened.

---

## 4. Files Changed

| File | Change Description |
| :--- | :--- |
| `frontend/package.json` | Added `tsx` to `devDependencies`, added `"test"` script. |
| `frontend/package-lock.json` | Recorded `tsx` dependency lockfile entry. |
| `.github/workflows/tests.yml` | Updated test commands to use `npx tsx` instead of `node`. |

---

## 5. Dependencies Changed

- **Added (devDependency)**: `tsx@^4.23.13`
- **Production Dependencies**: Unchanged (0 changes)

---

## 6. Local Test Results

All tests executed and verified locally:

### A. Candidate Metrics Selector & Switching Tests
```
$ npx tsx src/tests/test_candidate_metrics.mjs
--- Starting Candidate Metrics Selector & Switching Tests ---
✓ Candidate 1 extraction passed.
✓ Candidate 4 extraction and switching passed (latency: 9.818ms, throughput: 101.86, CPU: 68.2%).
✓ Strict metric isolation between candidates verified.
✓ ONNX candidate dynamic artifact detection passed.
--- ALL CANDIDATE METRICS TESTS PASSED ---
Exit Code: 0
```

### B. Candidate Normalization & Safety Regression Tests (Forms A-J)
```
$ npx tsx src/tests/test_candidate_normalization.mjs
--- Starting Candidate Normalization & Safety Regression Tests (Forms A-J) ---
✓ Form A (Complete Candidate Result) passed.
✓ Form B (Missing Optional Metric) passed.
✓ Form C (Null Metric) passed.
✓ Form D (Raw Numeric Metric) passed.
✓ Form E (ProvenanceMetric Metric) passed.
✓ Form F (Failed Candidate) passed.
✓ Form G (candidate_done Event) passed.
✓ Form H (best_candidate_selected Event) passed.
✓ Form I (Malformed Optional Field) passed.
✓ Form J (Empty / Null Candidate) passed.
--- ALL 10 CANDIDATE PAYLOAD REGRESSION TESTS PASSED (A to J) ---
Exit Code: 0
```

---

## 7. Frontend Build Result

```
$ npm run build
> frontend@0.0.0 build
> tsc -b && vite build

vite v8.2.2 building client environment for production...
transforming...
✓ 2437 modules transformed.
rendering chunks...
computing gzip size...
dist/index.html                   0.45 kB │ gzip:   0.29 kB
dist/assets/index-Bsk68MP3.css   56.64 kB │ gzip:   9.97 kB
dist/assets/index-DVDnoEf4.js   887.55 kB │ gzip: 238.28 kB
✓ built in 701ms
Exit Code: 0
```

---

## 8. Backend Verification Result

All backend CLI tests and regression suites executed with 100% pass rate:

1. **CLI Help & Arguments**:
   ```
   $ python uaqe.py --help
   usage: uaqe [-h] {optimize,plan,inspect-model,inspect-dataset,validate} ...
   Exit Code: 0
   ```
2. **Candidate Metric Identity Suite**:
   ```
   $ python pytest.py -q tests/test_candidate_metric_identity.py
   Ran 24 tests in 10.359s
   OK
   ```
3. **Candidate Latency Provenance Suite**:
   ```
   $ python pytest.py -q tests/test_candidate_latency_provenance.py
   Ran 15 tests in 0.007s
   OK
   ```
4. **Candidate Selection & Dynamic Download Suite**:
   ```
   $ python pytest.py -q tests/test_candidate_selection_and_download.py
   Ran 10 tests in 0.565s
   OK
   ```

---

## 9. GitHub Actions Result

- **Workflow File**: `.github/workflows/tests.yml`
- **Jobs**:
  - `backend-tests` (matrix: Python 3.11, 3.12): **PASS**
  - `frontend-build` (Node.js 20): **PASS**

---

## 10. Remaining Warnings

1. **Vite Bundle Size Warning**:
   ```
   (!) Some chunks are larger than 500 kB after minification. Consider:
   - Using dynamic import() to code-split the application
   ```
   *Note: This is standard Vite packaging advice for large vendor bundles (e.g., Recharts, Framer Motion) and does not affect runtime or build success.*

2. **GitHub Actions Node 20 Runtime Deprecation Notice**:
   GitHub Actions platform displays a platform-level deprecation notice regarding future runner default upgrades. It is informational only and does not impact build execution.
