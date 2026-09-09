import os, sys, json, hashlib, time, csv
import numpy as np
from pathlib import Path

PROJECT_ROOT  = Path(r"D:\Quantization embedded")
DATASET_ROOT  = Path(r"D:\semiconductor_dataset\dataset")
REAL_TEST_DIR = DATASET_ROOT / "test"
TRAIN_DIR     = DATASET_ROOT / "train"
VAL_DIR       = DATASET_ROOT / "val"
TFLITE_MODEL  = PROJECT_ROOT / "output" / "phase_c4" / "models" / "c4_best_int8.tflite"
FP32_ONNX     = PROJECT_ROOT / "output" / "phase_c3" / "models" / "c3_2_tailored_observers.onnx"
OUTPUT_DIR    = PROJECT_ROOT / "output" / "phase_c5"
REPORTS_DIR   = PROJECT_ROOT / "reports" / "phase_c5"
CLASS_NAMES   = sorted(["bridge","clean","cmp","crack","opens","other","particle","scratch","vias"])
CLASS_TO_IDX  = {c: i for i, c in enumerate(CLASS_NAMES)}
IMG_EXTS      = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
IMG_SIZE      = (128, 128)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_images_independent(split_dir):
    from PIL import Image as PILImage
    images, labels, paths = [], [], []
    for cls in CLASS_NAMES:
        cls_dir = split_dir / cls
        if not cls_dir.is_dir():
            continue
        idx = CLASS_TO_IDX[cls]
        for fname in sorted(cls_dir.iterdir()):
            if fname.suffix.lower() in IMG_EXTS:
                img = PILImage.open(fname).convert("RGB").resize(IMG_SIZE, PILImage.Resampling.BILINEAR)
                arr = np.array(img, dtype=np.float32) / 255.0
                images.append(arr.transpose(2, 0, 1))
                labels.append(idx)
                paths.append(str(fname))
    return np.stack(images).astype(np.float32), np.array(labels, dtype=np.int32), paths


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def v1_tflite_eval(x, y):
    print("\n" + "="*60)
    print("V1: INDEPENDENT TFLITE INT8 EVALUATION")
    print("="*60)
    import tensorflow as tf
    interp = tf.lite.Interpreter(
        model_path=str(TFLITE_MODEL),
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    )
    interp.allocate_tensors()
    in_det  = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    print(f"  Input : idx={in_det['index']}, shape={in_det['shape']}, dtype={in_det['dtype']}")
    print(f"  Output: idx={out_det['index']}, shape={out_det['shape']}, dtype={out_det['dtype']}")
    preds, logits_all = [], []
    t0 = time.perf_counter()
    for i in range(len(x)):
        interp.set_tensor(in_det["index"], x[i:i+1])
        interp.invoke()
        out = interp.get_tensor(out_det["index"])[0]
        logits_all.append(out.copy())
        preds.append(int(np.argmax(out)))
    elapsed = (time.perf_counter() - t0) * 1000.0
    preds  = np.array(preds)
    logits = np.array(logits_all)
    correct = int(np.sum(preds == y))
    acc = correct / len(y)
    per_class = {}
    for idx2, cls in enumerate(CLASS_NAMES):
        mask = (y == idx2)
        sup = int(np.sum(mask))
        c   = int(np.sum(preds[mask] == idx2)) if sup > 0 else 0
        per_class[cls] = {"support": sup, "correct": c, "accuracy": round(c/sup,4) if sup > 0 else None}
    pred_dist = {cls: int(np.sum(preds == i)) for i, cls in enumerate(CLASS_NAMES)}
    print(f"\n  N={len(x)} | Correct={correct} | Accuracy={acc*100:.4f}%")
    print(f"  Eval time: {elapsed:.1f} ms ({elapsed/len(x):.2f} ms/img)")
    print(f"\n  Per-class accuracy:")
    for cls, v in per_class.items():
        if v["accuracy"] is not None:
            print(f"    {cls:12s}: {v['correct']:3d}/{v['support']:3d} = {v['accuracy']*100:.1f}%")
    print(f"\n  Prediction distribution:")
    for cls, cnt in pred_dist.items():
        print(f"    {cls:12s}: {cnt:3d}  {'|'*cnt}")
    ls = logits
    print(f"\n  Logit stats: mean={np.mean(ls):.4f} std={np.std(ls):.4f} min={np.min(ls):.4f} max={np.max(ls):.4f}")
    result = {
        "n": len(x), "correct": correct, "accuracy": round(acc,6), "accuracy_pct": round(acc*100,4),
        "elapsed_ms": round(elapsed,2), "per_class": per_class, "pred_distribution": pred_dist,
        "logit_stats": {"mean": float(np.mean(ls)), "std": float(np.std(ls)),
                        "min": float(np.min(ls)), "max": float(np.max(ls))},
        "input_dtype": str(in_det["dtype"]), "output_dtype": str(out_det["dtype"])
    }
    return result, preds, logits


def v2_fp32_eval(x, y):
    print("\n" + "="*60)
    print("V2: INDEPENDENT FP32 ONNX REFERENCE EVALUATION")
    print("="*60)
    import onnxruntime as ort
    sess    = ort.InferenceSession(str(FP32_ONNX))
    in_name = sess.get_inputs()[0].name
    print(f"  ONNX input: {in_name}, shape={sess.get_inputs()[0].shape}")
    logits  = sess.run(None, {in_name: x})[0]
    preds   = np.argmax(logits, axis=1)
    correct = int(np.sum(preds == y))
    acc     = correct / len(y)
    per_class = {}
    for idx2, cls in enumerate(CLASS_NAMES):
        mask = (y == idx2)
        sup  = int(np.sum(mask))
        c    = int(np.sum(preds[mask] == idx2)) if sup > 0 else 0
        per_class[cls] = {"support": sup, "correct": c, "accuracy": round(c/sup,4) if sup > 0 else None}
    print(f"\n  N={len(x)} | Correct={correct} | Accuracy={acc*100:.4f}%")
    print(f"\n  Per-class accuracy:")
    for cls, v in per_class.items():
        if v["accuracy"] is not None:
            print(f"    {cls:12s}: {v['correct']:3d}/{v['support']:3d} = {v['accuracy']*100:.1f}%")
    return {"n": len(x), "correct": correct, "accuracy": round(acc,6), "accuracy_pct": round(acc*100,4),
            "per_class": per_class}, preds, logits


def v3_agreement(tflite_preds, fp32_preds, y):
    print("\n" + "="*60)
    print("V3: CROSS-PREDICTION AGREEMENT")
    print("="*60)
    n = len(y)
    agree = int(np.sum(tflite_preds == fp32_preds))
    disagree_idx = np.where(tflite_preds != fp32_preds)[0]
    print(f"  Agreement: {agree}/{n} = {agree/n*100:.4f}%")
    print(f"  Disagreements: {len(disagree_idx)}")
    if len(disagree_idx) > 0:
        print("  Disagreement details (idx | true | fp32 | tflite):")
        for di in disagree_idx[:20]:
            print(f"    [{di:4d}] true={CLASS_NAMES[y[di]]}, fp32={CLASS_NAMES[fp32_preds[di]]}, tflite={CLASS_NAMES[tflite_preds[di]]}")
    else:
        print("  All predictions agree perfectly.")
    return {"n": n, "agreements": agree, "agreement_rate": round(agree/n,6), "disagreements": len(disagree_idx), "disagree_indices": disagree_idx.tolist()}


def v4_logit_audit(tflite_logits, fp32_logits):
    print("\n" + "="*60)
    print("V4: LOGIT NUMERICAL FIDELITY AUDIT")
    print("="*60)
    from scipy.spatial.distance import cosine as sp_cosine
    tf_flat = tflite_logits.flatten().astype(float)
    fp_flat = fp32_logits.flatten().astype(float)
    cos  = float(1.0 - sp_cosine(tf_flat, fp_flat))
    mae  = float(np.mean(np.abs(tflite_logits - fp32_logits)))
    rmse = float(np.sqrt(np.mean((tflite_logits - fp32_logits)**2)))
    maxe = float(np.max(np.abs(tflite_logits - fp32_logits)))
    top1_agree = int(np.sum(np.argmax(tflite_logits,1) == np.argmax(fp32_logits,1)))
    fp32_norm   = float(np.linalg.norm(fp_flat))
    tflite_norm = float(np.linalg.norm(tf_flat))
    print(f"  Cosine similarity (full logit vectors): {cos:.6f}")
    print(f"  MAE vs FP32   : {mae:.6f}")
    print(f"  RMSE vs FP32  : {rmse:.6f}")
    print(f"  Max abs error : {maxe:.6f}")
    print(f"  Top-1 argmax agreement: {top1_agree}/{len(fp32_logits)} = {top1_agree/len(fp32_logits)*100:.2f}%")
    print(f"  FP32 L2-norm  : {fp32_norm:.4f}")
    print(f"  TFLite L2-norm: {tflite_norm:.4f}")
    print(f"  Scale ratio   : {tflite_norm/max(fp32_norm,1e-9):.4f}")
    return {"cosine_similarity": round(cos,6), "mae": round(mae,6), "rmse": round(rmse,6),
            "max_error": round(maxe,6), "top1_argmax_agree": top1_agree,
            "top1_agree_rate": round(top1_agree/len(fp32_logits),6),
            "fp32_l2": round(fp32_norm,4), "tflite_l2": round(tflite_norm,4),
            "scale_ratio": round(tflite_norm/max(fp32_norm,1e-9),4)}


def v5_leakage(test_paths):
    print("\n" + "="*60)
    print("V5: DATA LEAKAGE AUDIT (SHA-256)")
    print("="*60)
    print("  Hashing test images...")
    test_h = {sha256_file(p): p for p in test_paths}
    dups   = len(test_paths) - len(test_h)
    print(f"  {len(test_paths)} test images => {len(test_h)} unique hashes | intra-test dups: {dups}")
    print("  Hashing train images...")
    train_h = {}
    for cls in CLASS_NAMES:
        for f in sorted((TRAIN_DIR/cls).iterdir()):
            if f.suffix.lower() in IMG_EXTS:
                train_h[sha256_file(str(f))] = str(f)
    print(f"  {len(train_h)} train images hashed")
    overlap_train = {h: {"test": test_h[h], "train": train_h[h]} for h in test_h if h in train_h}
    if overlap_train:
        print(f"  !!! TRAIN LEAKAGE: {len(overlap_train)} test images in train set !!!")
        for h, info in list(overlap_train.items())[:5]:
            print(f"    test={os.path.basename(info['test'])} <-> train={os.path.basename(info['train'])}")
    else:
        print(f"  CLEAN: 0 test images found in train set")
    print("  Hashing val images...")
    val_h = {}
    for cls in CLASS_NAMES:
        for f in sorted((VAL_DIR/cls).iterdir()):
            if f.suffix.lower() in IMG_EXTS:
                val_h[sha256_file(str(f))] = str(f)
    overlap_val = {h: {"test": test_h[h], "val": val_h[h]} for h in test_h if h in val_h}
    if overlap_val:
        print(f"  !!! VAL LEAKAGE: {len(overlap_val)} test images in val set !!!")
    else:
        print(f"  CLEAN: 0 test images found in val set")
    verdict = "CLEAN" if not overlap_train and not overlap_val and dups == 0 else "CONTAMINATED"
    return {"n_test": len(test_paths), "unique_test": len(test_h), "n_train": len(train_h), "n_val": len(val_h),
            "train_overlap": len(overlap_train), "val_overlap": len(overlap_val), "intra_test_dups": dups,
            "verdict": verdict}


def v6_calib_report():
    print("\n" + "="*60)
    print("V6: CALIBRATION IMBALANCE ANALYSIS")
    print("="*60)
    bridge_count = sum(1 for f in sorted((TRAIN_DIR/"bridge").iterdir()) if f.suffix.lower() in IMG_EXTS)
    print(f"  Bridge images in train: {bridge_count} (first class alphabetically)")
    print(f"  C4-1 uses first 50 train samples = all bridge class")
    print(f"  C4 accuracy inversely correlated with calibration diversity:")
    rows = [("C4-1","50 (all bridge)","97.97%"),("C4-2","150 (bridge+clean)","88.32%"),
            ("C4-3","250 (bridge+clean+cmp)","86.29%"),("C4-4","100 (all bridge)","87.82%")]
    for exp,calib,acc in rows:
        print(f"    {exp}: {calib:30s} => {acc}")
    print("  FINDING: Bridge-only calibration paradoxically yields highest accuracy.")
    print("  This is the primary anomaly requiring explanation by independent eval.")
    return {"bridge_train": bridge_count, "c41_acc": 0.9797, "c42_acc": 0.8832,
            "c43_acc": 0.8629, "finding": "bridge-only calibration anomaly"}


def trust_classify(v1, v2, v3r, v4r, v5r):
    print("\n" + "="*60)
    print("TRUST CLASSIFICATION")
    print("="*60)
    issues, warnings = [], []
    if v5r["verdict"] != "CLEAN":
        issues.append(f"DATA_LEAKAGE: {v5r['train_overlap']} test images in train set")
    if abs(v1["accuracy_pct"] - 97.97) < 0.1:
        warnings.append(f"TFLite accuracy ({v1['accuracy_pct']:.2f}%) matches C4 claim exactly")
    if v4r["cosine_similarity"] < 0.5:
        warnings.append(f"Low logit cosine ({v4r['cosine_similarity']:.4f}) despite high accuracy (scale mismatch)")
    if v3r["agreement_rate"] > 0.99:
        warnings.append(f"Near-perfect FP32/TFLite agreement ({v3r['agreement_rate']*100:.2f}%) despite logit divergence")
    if issues:
        verdict = "UNVERIFIED_CRITICAL_ISSUES"
    elif len(warnings) >= 3:
        verdict = "CONDITIONALLY_VERIFIED_ANOMALIES_PRESENT"
    else:
        verdict = "VERIFIED"
    print(f"  TFLite accuracy   : {v1['accuracy_pct']:.4f}%")
    print(f"  FP32 accuracy     : {v2['accuracy_pct']:.4f}%")
    print(f"  FP32/TFLite agree : {v3r['agreement_rate']*100:.2f}%")
    print(f"  Logit cosine      : {v4r['cosine_similarity']:.6f}")
    print(f"  Leakage           : {v5r['verdict']}")
    print(f"  Issues  : {issues or 'None'}")
    print(f"  Warnings: {warnings or 'None'}")
    print(f"  VERDICT : {verdict}")
    return {"tflite_acc": v1["accuracy_pct"], "fp32_acc": v2["accuracy_pct"],
            "agreement_pct": v3r["agreement_rate"]*100, "cosine": v4r["cosine_similarity"],
            "leakage": v5r["verdict"], "issues": issues, "warnings": warnings, "verdict": verdict}


def main():
    print("="*60)
    print("  UAQE PHASE C.5 - INDEPENDENT VERIFICATION AUDIT")
    print("="*60)

    print("\n[LOAD] Loading test images from disk (independent)...")
    x, y, paths = load_images_independent(REAL_TEST_DIR)
    print(f"  {len(x)} images, shape={x.shape}, dtype={x.dtype}")
    for idx2, cls in enumerate(CLASS_NAMES):
        print(f"    {cls:12s}: {int(np.sum(y==idx2))}")

    v1r, tflite_preds, tflite_logits = v1_tflite_eval(x, y)
    v2r, fp32_preds,   fp32_logits   = v2_fp32_eval(x, y)
    v3r  = v3_agreement(tflite_preds, fp32_preds, y)
    v4r  = v4_logit_audit(tflite_logits, fp32_logits)
    v5r  = v5_leakage(paths)
    v6r  = v6_calib_report()
    trust = trust_classify(v1r, v2r, v3r, v4r, v5r)

    report = {
        "phase": "C.5", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "v1_tflite": v1r, "v2_fp32": v2r, "v3_agreement": v3r,
        "v4_logit": v4r, "v5_leakage": v5r, "v6_calib": v6r, "trust": trust
    }
    rp = OUTPUT_DIR / "c5_audit_report.json"
    with open(rp, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[SAVED] {rp}")

    pred_csv = OUTPUT_DIR / "c5_per_image_predictions.csv"
    with open(pred_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx","fname","true_cls","true_idx","fp32_pred","fp32_cls","fp32_ok","tflite_pred","tflite_cls","tflite_ok","agree"])
        for i,(p,ti) in enumerate(zip(paths, y)):
            w.writerow([i, os.path.basename(p), CLASS_NAMES[ti], ti,
                        int(fp32_preds[i]), CLASS_NAMES[int(fp32_preds[i])], int(fp32_preds[i]==ti),
                        int(tflite_preds[i]), CLASS_NAMES[int(tflite_preds[i])], int(tflite_preds[i]==ti),
                        int(fp32_preds[i]==tflite_preds[i])])
    print(f"[SAVED] {pred_csv}")

    md = REPORTS_DIR / "c5_verification_report.md"
    with open(md, "w") as f:
        f.write("# UAQE Phase C.5: Independent Verification Report\n\n")
        f.write(f"Timestamp: {report['timestamp']}\n\n---\n\n")
        f.write("## Summary\n\n")
        f.write(f"| Metric | C4 Claim | C5 Independent |\n|:---|:---:|:---:|\n")
        f.write(f"| TFLite INT8 Accuracy | 97.97% | **{v1r['accuracy_pct']:.4f}%** |\n")
        f.write(f"| FP32 ONNX Accuracy | 97.46% | **{v2r['accuracy_pct']:.4f}%** |\n")
        f.write(f"| FP32/TFLite Agreement | 100.00% | **{v3r['agreement_rate']*100:.4f}%** |\n")
        f.write(f"| Logit Cosine Similarity | N/A | **{v4r['cosine_similarity']:.6f}** |\n")
        f.write(f"| Data Leakage | N/A | **{v5r['verdict']}** |\n\n")
        f.write(f"## Final Verdict\n\n**{trust['verdict']}**\n\n")
        if trust['issues']:
            f.write("### Critical Issues\n")
            for iss in trust['issues']: f.write(f"- {iss}\n")
            f.write("\n")
        if trust['warnings']:
            f.write("### Warnings\n")
            for w2 in trust['warnings']: f.write(f"- {w2}\n")
            f.write("\n")
        f.write("## Per-Class Accuracy (TFLite INT8)\n\n")
        f.write("| Class | Support | Correct | Accuracy |\n|:---|:---:|:---:|:---:|\n")
        for cls, v in v1r["per_class"].items():
            acc_s = f"{v['accuracy']*100:.1f}%" if v['accuracy'] is not None else "N/A"
            f.write(f"| {cls} | {v['support']} | {v['correct']} | {acc_s} |\n")
        f.write("\n## V4 Logit Numerical Fidelity\n\n")
        f.write("| Metric | Value |\n|:---|:---:|\n")
        for k,v in v4r.items(): f.write(f"| {k} | {v} |\n")
    print(f"[SAVED] {md}")

    print("\n" + "="*60)
    print(f"  PHASE C.5 COMPLETE | VERDICT: {trust['verdict']}")
    print("="*60)


if __name__ == "__main__":
    main()
