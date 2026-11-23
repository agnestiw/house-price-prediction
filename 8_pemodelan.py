import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    make_scorer,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# ===========================================================================
# 1. KONFIGURASI
# ===========================================================================

SEED = 42
N_ITER = int(os.environ.get("N_ITER", 100))   # ubah lewat variabel lingkungan hanya untuk uji cepat
FOLDS = 5
N_BOOTSTRAP = 2000   # jumlah resampling untuk interval kepercayaan R² data uji

# Metrik untuk MEMILIH parameter saat tuning (bukan metrik laporan akhir):
#   "r2_log"    : R² pada skala log1p(harga) (pada uji saya tidak konsisten lebih baik)
#   "r2_rupiah" : R² pada skala Rupiah (perilaku skrip sebelumnya)
SCORING_TUNING = "r2_rupiah"

# Jumlah proses paralel untuk pencarian parameter.
# Gunakan 1 atau 2 jika penggunaan RAM terlalu besar.
N_JOBS_SEARCH = -1

INPUT_TRAIN = Path("output/data_train.csv")
INPUT_TEST = Path("output/data_test.csv")
OUT = Path("output/pemodelan")

TARGET = "harga"
GROUP = "property_id"

KOLOM_KATEGORIKAL = [
    "kecamatan",
    "Wilayah",
    "sertifikat",
    "KPR",
    "lebar jalan",
    "kondisi bangunan",
    "tipe lingkungan",
]

# Harus sama dengan pemetaan pada 4_preprocessing.py.
ENCODING_FITUR_BARU = {
    "kondisi bangunan": {
        "tidak diketahui": 0,
        "lama": 1,
        "baru": 2,
    },
    "tipe lingkungan": {
        "tidak diketahui": 0,
        "kampung": 1,
        "perumahan": 2,
    },
}

NAMA_FILE_MODEL = {
    "Random Forest": "model_random_forest.pkl",
    "XGBoost": "model_xgboost.pkl",
    "LightGBM": "model_lightgbm.pkl",
}


# ===========================================================================
# 2. FUNGSI BANTU
# ===========================================================================

def judul(teks):
    print("\n" + "=" * 75)
    print(teks)
    print("=" * 75)


def hitung_metrik(aktual, prediksi):
    aktual = np.asarray(aktual, dtype=float)
    prediksi = np.asarray(prediksi, dtype=float)

    if not np.isfinite(prediksi).all():
        raise ValueError("Prediksi mengandung NaN atau nilai tidak finite.")

    return {
        "R2": float(r2_score(aktual, prediksi)),
        "R2_log": float(
            r2_score(np.log1p(aktual), np.log1p(np.clip(prediksi, 0, None)))
        ),
        "MAE": float(mean_absolute_error(aktual, prediksi)),
        "RMSE": float(np.sqrt(mean_squared_error(aktual, prediksi))),
        "MAPE": float(
            100 * mean_absolute_percentage_error(aktual, prediksi)
        ),
    }


def tabel_prediksi(ids, aktual, prediksi, fold=None):
    aktual = np.asarray(aktual, dtype=float)
    prediksi = np.asarray(prediksi, dtype=float)

    hasil = pd.DataFrame({
        "property_id": np.asarray(ids),
        "aktual": aktual,
        "prediksi": prediksi,
        "residual": aktual - prediksi,
        "error_absolut": np.abs(aktual - prediksi),
        "error_kuadrat": np.square(aktual - prediksi),
        "error_persen": 100 * np.abs(aktual - prediksi) / aktual,
    })

    if fold is not None:
        hasil.insert(0, "fold", fold)

    return hasil


def r2_skala_log(aktual, prediksi):
    """R² pada skala log1p(harga); prediksi model tetap dalam Rupiah."""
    return r2_score(
        np.log1p(np.asarray(aktual, dtype=float)),
        np.log1p(np.clip(np.asarray(prediksi, dtype=float), 0, None)),
    )


def pilih_scoring():
    if SCORING_TUNING == "r2_log":
        return make_scorer(r2_skala_log)
    if SCORING_TUNING == "r2_rupiah":
        return "r2"
    raise ValueError("SCORING_TUNING harus 'r2_log' atau 'r2_rupiah'.")


def interval_bootstrap_r2(aktual, prediksi, n=N_BOOTSTRAP, seed=SEED):
    """Interval kepercayaan 95% untuk R² data uji (resampling baris data uji)."""
    aktual = np.asarray(aktual, dtype=float)
    prediksi = np.asarray(prediksi, dtype=float)
    rng = np.random.default_rng(seed)
    nilai = []

    for _ in range(n):
        idx = rng.integers(0, len(aktual), len(aktual))
        nilai.append(r2_score(aktual[idx], prediksi[idx]))

    bawah, atas = np.percentile(nilai, [2.5, 97.5])
    return float(bawah), float(atas)


def buat_search(pipeline, ruang_parameter, splits):
    return RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=ruang_parameter,
        n_iter=N_ITER,
        scoring=pilih_scoring(),
        cv=splits,
        refit=True,
        random_state=SEED,
        n_jobs=N_JOBS_SEARCH,
        error_score="raise",
        return_train_score=True,
    )


# ===========================================================================
# 3. PEMBACAAN DAN VALIDASI DATA
# ===========================================================================

def baca_data():
    train = pd.read_csv(INPUT_TRAIN, dtype={GROUP: "string"})
    test = pd.read_csv(INPUT_TEST, dtype={GROUP: "string"})

    wajib = {TARGET, GROUP, *ENCODING_FITUR_BARU}

    for label, data in [("Training", train), ("Testing", test)]:
        hilang = wajib - set(data.columns)

        if hilang:
            raise ValueError(
                f"{label}: kolom wajib belum tersedia: {sorted(hilang)}. "
                "Jalankan ulang mapping, merge, preprocessing, dan data split."
            )

        if data.empty:
            raise ValueError(f"Data {label} kosong.")

        ids = data[GROUP].str.strip()

        if ids.isna().any() or ids.eq("").any():
            raise ValueError(f"{label}: property_id tidak boleh kosong.")

        data[GROUP] = ids

        data[TARGET] = pd.to_numeric(data[TARGET], errors="raise")

        target = data[TARGET].to_numpy(dtype=float)

        if not np.isfinite(target).all() or (target <= 0).any():
            raise ValueError(
                f"{label}: harga harus positif, tidak kosong, dan finite."
            )

    features = [
        kol for kol in train.columns
        if kol not in [TARGET, GROUP]
    ]

    fitur_test = [
        kol for kol in test.columns
        if kol not in [TARGET, GROUP]
    ]

    if set(features) != set(fitur_test):
        raise ValueError(
            "Kolom fitur training dan testing tidak sama.\n"
            f"Hanya di training: {sorted(set(features) - set(fitur_test))}\n"
            f"Hanya di testing: {sorted(set(fitur_test) - set(features))}"
        )

    categorical = [
        kol for kol in KOLOM_KATEGORIKAL if kol in features
    ]
    numeric = [
        kol for kol in features if kol not in categorical
    ]

    # Input harus sudah melalui preprocessing dan encoding.
    # Jangan mengganti kode kategori 0 menjadi NaN.
    for label, data in [("Training", train), ("Testing", test)]:
        for kol in features:
            try:
                data[kol] = pd.to_numeric(data[kol], errors="raise")
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{label}: kolom '{kol}' masih berisi teks. "
                    "Gunakan hasil preprocessing yang sudah di-encoding."
                ) from exc

            nilai = data[kol].to_numpy(dtype=float)

            if np.isinf(nilai).any():
                raise ValueError(
                    f"{label}: kolom '{kol}' mengandung infinity."
                )

        for kol, pemetaan in ENCODING_FITUR_BARU.items():
            # Nilai kosong pada fitur baru berarti tidak diketahui.
            data[kol] = data[kol].fillna(pemetaan["tidak diketahui"])

            kode_valid = set(pemetaan.values())
            tidak_valid = ~data[kol].isin(kode_valid)

            if tidak_valid.any():
                contoh = data.loc[tidak_valid, kol].unique().tolist()
                raise ValueError(
                    f"{label}: kode '{kol}' tidak valid: {contoh}. "
                    f"Kode yang diperbolehkan: {sorted(kode_valid)}."
                )

            data[kol] = data[kol].astype(int)

    irisan = set(train[GROUP]) & set(test[GROUP])

    if irisan:
        raise ValueError(
            f"Ada {len(irisan)} property_id yang muncul di training "
            "dan testing. Perbaiki pembagian data terlebih dahulu."
        )

    if train[GROUP].nunique() < FOLDS:
        raise ValueError(
            f"Jumlah property_id unik pada training harus minimal {FOLDS}."
        )

    return train, test, features, categorical, numeric


# ===========================================================================
# 4. KONFIGURASI MODEL DAN RUANG PARAMETER
# ===========================================================================

def konfigurasi_model():
    return {
        "Random Forest": (
            RandomForestRegressor(
                random_state=SEED,
                n_jobs=1,
            ),
            {
                "n_estimators": [300, 500],
                "max_depth": [6, 10, 15, 20],
                "min_samples_leaf": [2, 4, 8, 12],
                "max_features": [0.5, 0.8, 1.0],
            },
        ),
        "XGBoost": (
            XGBRegressor(
                objective="reg:squarederror",
                random_state=SEED,
                n_jobs=1,
            ),
            {
                "n_estimators": [200, 500, 800],
                "max_depth": [3, 4, 6],
                "learning_rate": [0.03, 0.05, 0.1],
                "min_child_weight": [3, 5, 10],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
                "reg_lambda": [1, 5, 10],
                "reg_alpha": [0, 0.1, 1],
            },
        ),
        "LightGBM": (
            LGBMRegressor(
                random_state=SEED,
                n_jobs=1,
                verbosity=-1,
                subsample_freq=1,
            ),
            {
                "n_estimators": [200, 500, 800],
                "num_leaves": [15, 31],
                "max_depth": [5, 7, 9],
                "min_child_samples": [10, 20, 40],
                "learning_rate": [0.03, 0.05, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
                "reg_lambda": [0, 1, 5],
            },
        ),
    }


def konfigurasi_baseline():
    """Model dengan parameter bawaan pustaka, sebagai pembanding sebelum tuning."""
    return {
        "Random Forest": RandomForestRegressor(random_state=SEED, n_jobs=1),
        "XGBoost": XGBRegressor(
            objective="reg:squarederror",
            random_state=SEED,
            n_jobs=1,
        ),
        "LightGBM": LGBMRegressor(
            random_state=SEED,
            n_jobs=1,
            verbosity=-1,
        ),
    }


# ===========================================================================
# 5. PROSES UTAMA
# ===========================================================================

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    train, test, features, categorical, numeric = baca_data()

    X = train[features].copy()
    y = train[TARGET].copy()
    groups = train[GROUP].copy()

    Xt = test[features].copy()
    yt = test[TARGET].copy()

    judul("1. DATA DAN FITUR")

    print(f"Training : {len(train):,} baris")
    print(f"Testing  : {len(test):,} baris")
    print(f"Fitur    : {len(features)}")
    print(f"Kategori : {categorical}")
    print(f"Numerik  : {numeric}")
    print(f"Tuning   : {N_ITER} kombinasi × {FOLDS} inner fold")
    print(f"Evaluasi : {FOLDS} outer fold")
    print("Target   : log1p saat training; prediksi dalam Rupiah")

    # Median dan modus hanya dipelajari dari bagian training
    # pada setiap proses fit, termasuk di dalam cross-validation.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline([
                    (
                        "fill",
                        SimpleImputer(strategy="most_frequent"),
                    ),
                    (
                        "encode",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                        ),
                    ),
                ]),
                categorical,
            ),
        ],
        remainder="drop",
    )

    # Splits yang sama digunakan oleh ketiga algoritma.
    outer_splits = list(
        GroupKFold(n_splits=FOLDS).split(X, y, groups)
    )

    inner_splits_per_fold = []

    for fold, (idx_train, idx_valid) in enumerate(outer_splits, start=1):
        groups_inner = groups.iloc[idx_train]

        if groups_inner.nunique() < FOLDS:
            raise ValueError(
                f"Outer fold {fold}: jumlah grup training tidak cukup "
                f"untuk {FOLDS} inner fold."
            )

        # Indeks inner relatif terhadap subset training outer.
        inner_splits = list(
            GroupKFold(n_splits=FOLDS).split(
                X.iloc[idx_train],
                y.iloc[idx_train],
                groups_inner,
            )
        )

        inner_splits_per_fold.append(inner_splits)

    # Simpan keanggotaan outer fold untuk penelusuran.
    fold_assignment = np.zeros(len(train), dtype=int)

    for fold, (_, idx_valid) in enumerate(outer_splits, start=1):
        fold_assignment[idx_valid] = fold

    pd.DataFrame({
        "property_id": groups.to_numpy(),
        "harga": y.to_numpy(),
        "outer_fold": fold_assignment,
    }).to_csv(
        OUT / "pembagian_outer_fold.csv",
        index=False,
    )

    results = []
    cv_summary = []
    baseline_rows = []
    ci_test = {}

    for name, (estimator, params) in konfigurasi_model().items():
        judul(f"MODEL: {name}")

        pipeline = Pipeline([
            ("pre", clone(preprocessor)),
            (
                "model",
                TransformedTargetRegressor(
                    regressor=estimator,
                    func=np.log1p,
                    inverse_func=np.expm1,
                ),
            ),
        ])

        parameter_space = {
            f"model__regressor__{key}": values
            for key, values in params.items()
        }

        fold_rows = []
        oof_predictions = np.full(len(X), np.nan)

        # ---------------------------------------------------------------
        # A. NESTED CROSS-VALIDATION
        # ---------------------------------------------------------------

        for fold, (idx_train, idx_valid) in enumerate(
            outer_splits,
            start=1,
        ):
            print(
                f"\n{name} | outer fold {fold}/{FOLDS}: "
                "mencari parameter...",
                flush=True,
            )

            search = buat_search(
                pipeline,
                parameter_space,
                inner_splits_per_fold[fold - 1],
            )

            search.fit(
                X.iloc[idx_train],
                y.iloc[idx_train],
            )

            # Outer validation tidak digunakan dalam inner tuning.
            pred_train = search.predict(X.iloc[idx_train])
            pred_valid = search.predict(X.iloc[idx_valid])

            oof_predictions[idx_valid] = pred_valid

            metrics_train = hitung_metrik(
                y.iloc[idx_train],
                pred_train,
            )
            metrics_valid = hitung_metrik(
                y.iloc[idx_valid],
                pred_valid,
            )

            row = {
                "fold": fold,
                "jumlah_training": len(idx_train),
                "jumlah_validasi": len(idx_valid),
                "R2_training": metrics_train["R2"],
                **metrics_valid,
                "gap_R2": (
                    metrics_train["R2"] - metrics_valid["R2"]
                ),
            }

            fold_rows.append(row)

            print(
                f"{name} | fold {fold} | "
                f"R² train: {row['R2_training']:.4f} | "
                f"R² validasi: {row['R2']:.4f} | "
                f"RMSE: Rp {row['RMSE']:,.0f}",
                flush=True,
            )

            # Simpan semua prediksi fold, terbesar error-nya di atas.
            detail = tabel_prediksi(
                groups.iloc[idx_valid],
                y.iloc[idx_valid],
                pred_valid,
                fold=fold,
            )

            detail.sort_values(
                "error_kuadrat",
                ascending=False,
            ).to_csv(
                OUT / f"{name}_fold_{fold}_prediksi.csv",
                index=False,
            )

        folds = pd.DataFrame(fold_rows)

        folds.to_csv(
            OUT / f"{name}_nested_cv.csv",
            index=False,
        )

        # Setiap baris training diprediksi satu kali ketika menjadi
        # outer validation: prediksi out-of-fold (OOF).
        oof_metrics = hitung_metrik(y, oof_predictions)

        tabel_prediksi(
            groups,
            y,
            oof_predictions,
            fold=fold_assignment,
        ).to_csv(
            OUT / f"{name}_oof_prediksi.csv",
            index=False,
        )

        cv_summary.append({
            "Model": name,
            "R2_CV": folds["R2"].mean(),
            "R2_std": folds["R2"].std(),
            "R2_min": folds["R2"].min(),
            "R2_max": folds["R2"].max(),
            "R2_OOF": oof_metrics["R2"],
            "RMSE_OOF": oof_metrics["RMSE"],
        })

        # ---------------------------------------------------------------
        # B. TUNING FINAL PADA SELURUH DATA TRAINING
        # ---------------------------------------------------------------

        print(
            f"\n{name}: tuning final pada seluruh data training...",
            flush=True,
        )

        final_search = buat_search(
            pipeline,
            parameter_space,
            outer_splits,
        )

        final_search.fit(X, y)

        model = final_search.best_estimator_

        print("\nParameter terbaik:")

        for key, value in final_search.best_params_.items():
            nama_parameter = key.replace("model__regressor__", "")
            print(f"  {nama_parameter:<20}: {value}")

        print(
            "R² CV terbaik untuk pemilihan parameter final: "
            f"{final_search.best_score_:.4f}"
        )
        print(
            "Catatan: skor pemilihan parameter ini bukan "
            "pengganti hasil nested CV."
        )

        pd.DataFrame(final_search.cv_results_).to_csv(
            OUT / f"{name}_tuning.csv",
            index=False,
        )

        # ---------------------------------------------------------------
        # C. SIMPAN SATU BUNDLE UNTUK SETIAP ALGORITMA
        # ---------------------------------------------------------------

        bundle = {
            "model": model,
            "nama_model": name,
            "fitur": features,
            "fitur_kategorikal": categorical,
            "fitur_numerik": numeric,
            "encoding_fitur_baru": ENCODING_FITUR_BARU,
            "log_target": False,
            "target_transform_internal": "log1p/expm1",
            "input": (
                "fitur hasil preprocessing dan encoding "
                "sebelum normalisasi"
            ),
            "output": "harga dalam Rupiah",
            "parameter_terbaik": final_search.best_params_,
            "seed": SEED,
        }

        model_path = OUT / NAMA_FILE_MODEL[name]

        joblib.dump(bundle, model_path)

        # Pastikan model setelah dimuat menghasilkan prediksi yang sama.
        loaded = joblib.load(model_path)
        contoh_input = X.head(5)

        np.testing.assert_allclose(
            loaded["model"].predict(contoh_input),
            model.predict(contoh_input),
        )

        print(
            f"Model tersimpan dan diverifikasi: {model_path}",
            flush=True,
        )

        # ---------------------------------------------------------------
        # D. EVALUASI MODEL FINAL
        # ---------------------------------------------------------------

        for label, xx, yy in [
            ("Training", X, y),
            ("Testing", Xt, yt),
        ]:
            pred = model.predict(xx)

            results.append({
                "Model": name,
                "Data": label,
                **hitung_metrik(yy, pred),
            })

            if label == "Testing":
                ci_test[name] = interval_bootstrap_r2(yy, pred)

                tabel_prediksi(
                    test[GROUP],
                    yy,
                    pred,
                ).sort_values(
                    "error_kuadrat",
                    ascending=False,
                ).to_csv(
                    OUT / f"{name}_prediksi.csv",
                    index=False,
                )

        # ---------------------------------------------------------------
        # E. BASELINE: PARAMETER BAWAAN PUSTAKA (PEMBANDING SEBELUM TUNING)
        # Preprocessing, transformasi target, dan outer fold sama dengan model tuned.
        # ---------------------------------------------------------------

        baseline = Pipeline([
            ("pre", clone(preprocessor)),
            (
                "model",
                TransformedTargetRegressor(
                    regressor=konfigurasi_baseline()[name],
                    func=np.log1p,
                    inverse_func=np.expm1,
                ),
            ),
        ])

        oof_baseline = np.full(len(X), np.nan)

        for idx_train, idx_valid in outer_splits:
            model_fold = clone(baseline)
            model_fold.fit(X.iloc[idx_train], y.iloc[idx_train])
            oof_baseline[idx_valid] = model_fold.predict(X.iloc[idx_valid])

        oof_base = hitung_metrik(y, oof_baseline)

        baseline.fit(X, y)
        test_base = hitung_metrik(yt, baseline.predict(Xt))

        baseline_rows.append({
            "Model": name,
            "R2_OOF_baseline": oof_base["R2"],
            "R2_log_OOF_baseline": oof_base["R2_log"],
            "R2_test_baseline": test_base["R2"],
            "R2_log_test_baseline": test_base["R2_log"],
            "MAE_test_baseline": test_base["MAE"],
            "RMSE_test_baseline": test_base["RMSE"],
        })

    # ===================================================================
    # 6. RINGKASAN SELURUH MODEL
    # ===================================================================

    summary = pd.DataFrame(cv_summary).sort_values(
        "R2_CV",
        ascending=False,
    )
    evaluation = pd.DataFrame(results)

    summary.to_csv(
        OUT / "ringkasan_nested_cv.csv",
        index=False,
    )
    evaluation.to_csv(
        OUT / "evaluasi.csv",
        index=False,
    )

    judul("RINGKASAN NESTED CROSS-VALIDATION")

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print(
        "\nR2_CV adalah rata-rata R² antar-outer-fold."
        "\nR2_OOF dihitung dari seluruh prediksi outer validation "
        "yang digabungkan; nilainya dapat berbeda dari R2_CV."
    )

    judul("EVALUASI MODEL FINAL")

    print(
        evaluation.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    judul("TABEL EVALUASI UTAMA (DATA UJI, MODEL SETELAH TUNING)")

    uji = evaluation[evaluation["Data"] == "Testing"].copy()
    uji["R2_CI95_bawah"] = uji["Model"].map(lambda m: ci_test[m][0])
    uji["R2_CI95_atas"] = uji["Model"].map(lambda m: ci_test[m][1])
    utama = uji[[
        "Model", "R2", "MAE", "RMSE", "MAPE",
        "R2_log", "R2_CI95_bawah", "R2_CI95_atas",
    ]]
    utama.to_csv(OUT / "evaluasi_utama.csv", index=False)

    print(
        utama.to_string(
            index=False,
            float_format=lambda value: f"{value:,.4f}",
        )
    )
    print(
        "\nR2, MAE, RMSE, MAPE dihitung pada data uji dalam Rupiah."
        "\nR2_log = R² pada skala log1p(harga) (metrik tambahan)."
        "\nR2_CI95 = interval kepercayaan 95% (bootstrap) untuk R2 data uji."
    )

    judul("PERBANDINGAN BASELINE (PARAMETER BAWAAN) DAN TUNED")

    banding = pd.DataFrame(baseline_rows).merge(
        summary[["Model", "R2_OOF"]].rename(columns={"R2_OOF": "R2_OOF_tuned"}),
        on="Model",
    ).merge(
        uji[["Model", "R2", "R2_log"]].rename(columns={
            "R2": "R2_test_tuned",
            "R2_log": "R2_log_test_tuned",
        }),
        on="Model",
    )
    banding["selisih_R2_OOF"] = (
        banding["R2_OOF_tuned"] - banding["R2_OOF_baseline"]
    )
    banding["selisih_R2_test"] = (
        banding["R2_test_tuned"] - banding["R2_test_baseline"]
    )
    banding["selisih_R2_log_test"] = (
        banding["R2_log_test_tuned"] - banding["R2_log_test_baseline"]
    )
    banding.to_csv(OUT / "perbandingan_baseline_tuned.csv", index=False)

    print(
        banding[[
            "Model",
            "R2_OOF_baseline", "R2_OOF_tuned", "selisih_R2_OOF",
            "R2_test_baseline", "R2_test_tuned", "selisih_R2_test",
            "R2_log_test_baseline", "R2_log_test_tuned", "selisih_R2_log_test",
        ]].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print(
        "\nSelisih positif berarti tuning memperbaiki model terhadap parameter bawaan."
        "\nR2_OOF berasal dari nested CV (lebih stabil); R2_test dari satu pembagian data uji."
    )

    judul("SELISIH R² TRAINING DAN TESTING")

    for name in NAMA_FILE_MODEL:
        nilai = evaluation[evaluation["Model"] == name].set_index("Data")

        gap = (
            nilai.loc["Training", "R2"]
            - nilai.loc["Testing", "R2"]
        )

        print(f"{name:<15}: {gap:.6f}")

    judul("FILE MODEL UNTUK SISTEM")

    for filename in NAMA_FILE_MODEL.values():
        print(OUT / filename)

    print(
        "\nKetiga model disimpan terpisah."
        "\nGunakan bundle['model'].predict(data_input) untuk "
        "mendapatkan harga dalam Rupiah."
        "\nJangan melakukan inverse log lagi pada hasil prediksi."
    )


if __name__ == "__main__":
    main()