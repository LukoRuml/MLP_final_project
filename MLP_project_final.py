# Imports
import json, os, time
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Fragments
from rdkit.Chem.EState.EState import EStateIndices
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator

from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, StratifiedGroupKFold,
    GridSearchCV, HalvingRandomSearchCV, cross_val_predict,
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, accuracy_score,
    matthews_corrcoef, f1_score, precision_score, recall_score,
    fbeta_score, confusion_matrix, classification_report, roc_curve,
)

import lightgbm as lgb
import logging
import warnings

# Constants
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

warnings.filterwarnings(
    "ignore",
    message=".*sklearn.utils.parallel.delayed.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*does not have valid feature names.*",
    category=UserWarning,
)

# Basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def safe_roc_auc(y_true, y_score):
    """
    Return roc_auc_score or np.nan if it cannot be computed.

    This guards against degenerate label sets (single-class) which raise.
    """
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return float("nan")


# Encapsulate script logic
def main():
    # Data loading - activities for NAMPT from ChEMBL – treated for failiures and completion
    TARGET_ID = "CHEMBL1744525"
    BASE_URL  = "https://www.ebi.ac.uk/chembl/api/data"
    BATCH     = 1000
    CACHE     = f"chembl_{TARGET_ID}.json"
    MIN_EXPECTED = 4500  # expected minimal number of records

    def fetch_chembl(target_id):
        """
        Downloads bioactivity data for a specific target from the ChEMBL API.

        Args:
            target_id (str): The ChEMBL ID of the biological target (e.g., "CHEMBL1744525").

        Returns:
            list: A list of dictionaries, where each dictionary contains
                  activity data for a single compound.

        Raises:
            RuntimeError: If the API request fails to resolve after all retries.
        """
        activities, offset = [], 0
        while True:
            params = {
                "target_chembl_id": target_id,
                "pchembl_value__gte": 0,
                "limit": BATCH,
                "offset": offset,
            }
            for attempt in range(3):
                try:
                    resp = requests.get(f"{BASE_URL}/activity.json",
                                        params=params, timeout=120)
                    resp.raise_for_status()
                    payload = resp.json()
                    break
                except (requests.RequestException, ValueError) as exc:
                    wait = 2 ** attempt * 5  # 5s, 10s, 20s
                    logging.warning("  [retry %d/3] offset=%d error: %s; Waiting %ds",
                                    attempt+1, offset, exc, wait)
                    time.sleep(wait)
            else:
                raise RuntimeError(
                    f"ChEMBL API failed at offset={offset}. "
                    f"Downloaded {len(activities)} activities."
                )

            rows = payload.get("activities", [])
            if not rows:
                break
            activities.extend(rows)
            offset += BATCH
            logging.info("  ... offset=%5d, %5d total records", offset, len(activities))
        return activities

    # Cache route
    if os.path.exists(CACHE):
        logging.info("Loading from cache %s", CACHE)
        with open(CACHE, "r", encoding="utf-8") as f:
            activities = json.load(f)
    else:
        activities = fetch_chembl(TARGET_ID)
        with open(CACHE, "w", encoding="utf-8") as f:
            json.dump(activities, f)
        logging.info("Cache recorded into %s", CACHE)

    logging.info("Downloaded %d activities for %s", len(activities), TARGET_ID)

    # Incomplete data download
    if len(activities) < MIN_EXPECTED:
        raise RuntimeError(
            f"DOWNLOAD INCOMPLETE: {len(activities)} < {MIN_EXPECTED} activities. "
            f"ChEMBL failed (500 error). Delete {CACHE} and try again., "
            f"alternatively lower MIN_EXPECTED threshold."
        )

    # Data filtration
    df = pd.DataFrame(activities)
    df = df[df["standard_type"].isin(["IC50", "Ki", "EC50"])]
    df = df[df["standard_units"].isin(["nM", "uM", "M"])]
    df = df[df["pchembl_value"].notna()].copy()

    UNIT_TO_NM = {"nM": 1.0, "uM": 1e3, "M": 1e9}
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df["standard_value_nM"] = df["standard_value"] * df["standard_units"].map(UNIT_TO_NM)
    df["pIC50"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
    df = df.dropna(subset=["pIC50", "canonical_smiles"])

    # Activity threshold - 1 µM == pIC50 6.0
    ACTIVITY_THRESHOLD = 6.0
    df["active"] = (df["pIC50"] >= ACTIVITY_THRESHOLD).astype(int)

    # Blank columns cleaning
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    MIN_NON_NULL_RATIO = 0.10
    ALWAYS_KEEP = {
        "canonical_smiles", "standard_type", "standard_units", "standard_value",
        "standard_value_nM", "pchembl_value", "pIC50", "active",
        "molecule_chembl_id", "target_chembl_id", "target_pref_name",
    }
    non_null_ratio = df.notna().mean()
    cols_to_drop = sorted({
        c for c, r in non_null_ratio.items()
        if r < MIN_NON_NULL_RATIO and c not in ALWAYS_KEEP
    })
    df = df.drop(columns=cols_to_drop)

    # Redundant data cleaning
    META_DROP = [
        "activity_comment", "activity_properties", "assay_description", "bao_label",
        "assay_type", "bao_format", "bao_endpoint", "document_chembl_id",
        "document_journal", "document_year", "relation", "standard_relation",
        "target_tax_id", "parent_molecule_chembl_id", "potential_duplicate",
        "record_id", "qudt_units", "target_chembl_id", "target_organism",
        "target_pref_name", "uo_units", "type", "units", "value",
        "standard_units", "standard_value", "standard_flag", "assay_chembl_id", "src_id",
    ]
    df = df.drop(columns=META_DROP, errors="ignore")

    logging.info("After filtration: %d records", len(df))
    logging.info("  Active (pIC50 ≥ %.1f): %d", ACTIVITY_THRESHOLD, int(df['active'].sum()))
    logging.info("  Inactive: %d", int((df['active'] == 0).sum()))

    # Data record deduplication
    logging.info("Pre-deduplication: %d records (%d unique molecules)",
                 len(df), df['molecule_chembl_id'].nunique())

    df_agg = (
        df.groupby("molecule_chembl_id")
          .agg(pIC50=("pIC50", "median"),
               canonical_smiles=("canonical_smiles", "first")
               )
          .reset_index()
    )
    df_agg["active"] = (df_agg["pIC50"] >= ACTIVITY_THRESHOLD).astype(int)

    logging.info("After deduplication: %d unique molecules", len(df_agg))
    logging.info("Distribution of pIC50:\n%s", df_agg["pIC50"].describe().round(2))

    # Morgan FingerPrint Descriptors

    FP_BITS, FP_RADIUS = 2048, 2
    _morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)

    def morgan_fp(mol) -> np.ndarray:
        """
        Generates a Morgan fingerprint (ECFP) for a given molecular structure.

        Args:
            mol (rdkit.Chem.rdchem.Mol): The parsed RDKit molecule object.

        Returns:
            numpy.ndarray: A 1D array representing the binary fingerprint.
                   Returns an array of NaNs if the input molecule is invalid.
        """
        if mol is None:
            return np.full(FP_BITS, np.nan)
        return _morgan_gen.GetFingerprintAsNumPy(mol).astype(np.uint8)

    # Physchem Data
    PHYS_FUNCS = {
        "LogP":          Descriptors.MolLogP,
        "TPSA":          Descriptors.TPSA,
        "RotBonds":      Descriptors.NumRotatableBonds,
        "HeavyAtoms":    Descriptors.HeavyAtomCount,
        "FracArom":      lambda m: (Descriptors.NumAromaticRings(m) /
                                    max(1, Descriptors.RingCount(m))),
        "MolMR":         Descriptors.MolMR,
        "BalabanJ":      Descriptors.BalabanJ,
        "BertzCT":       Descriptors.BertzCT,
        "Chi0v":         Descriptors.Chi0v,
        "Kappa1":        Descriptors.Kappa1,
        "HallKierAlpha": Descriptors.HallKierAlpha,
        "NumHetero":     Descriptors.NumHeteroatoms,
        "NumNHOH":       Descriptors.NHOHCount,
        "qed":           Descriptors.qed,
    }

    # EState
    def _estate_stats(mol) -> list:
        """
        Computes summary statistics for the Electrotopological State (E-State) indices.

        E-State indices quantify the electronic and topological environment of each
        atom in the molecule. This function aggregates those individual atomic values
        to provide a whole-molecule descriptor.

        Args:
            mol (rdkit.Chem.rdchem.Mol): The parsed RDKit molecule object.

        Returns:
            list: A list containing four floats representing the [maximum, minimum,
                  mean, sum] of the E-State indices. Returns a list of four NaNs if
                  the calculation fails or the molecule has no valid indices.
        """
        try:
            idx = EStateIndices(mol)
            if len(idx) == 0:
                return [np.nan, np.nan, np.nan, np.nan]
            return [float(np.max(idx)), float(np.min(idx)),
                    float(np.mean(idx)), float(np.sum(idx))]
        except Exception:
            return [np.nan, np.nan, np.nan, np.nan]

    # Molecule Structure Fragments
    FRAG_FUNCS = {
        "fr_NH1":         Fragments.fr_NH1,
        "fr_ether":       Fragments.fr_ether,
        "fr_pyridine":    Fragments.fr_pyridine,
        "fr_piperdine":   Fragments.fr_piperdine,
    }

    ESTATE_NAMES = ["EState_max", "EState_min", "EState_mean", "EState_sum"]

    def physchem(mol) -> list:
        """
        Calculates a comprehensive set of 24 molecular descriptors including
        physicochemical properties, E-State statistics, and structural fragments.

        Args:
            mol (rdkit.Chem.rdchem.Mol): The parsed RDKit molecule object.

        Returns:
            list: A list of float values corresponding to the calculated descriptors
                  in a fixed, pre-determined order. If the input molecule is invalid,
                  returns a list of NaNs of the exact same length to maintain array geometry.
        """
        if mol is None:
            n = len(PHYS_FUNCS) + len(ESTATE_NAMES) + len(FRAG_FUNCS)
            return [np.nan] * n
        out = []
        for fn in PHYS_FUNCS.values():
            try:
                v = fn(mol)
                out.append(float(v) if v is not None else np.nan)
            except Exception:
                out.append(np.nan)
        out.extend(_estate_stats(mol))
        for fn in FRAG_FUNCS.values():
            try:
                out.append(float(fn(mol)))
            except Exception:
                out.append(np.nan)
        return out

    DESC_COLS = (list(PHYS_FUNCS.keys()) + ESTATE_NAMES + list(FRAG_FUNCS.keys()))
    N_DESC = len(DESC_COLS)

    fp_list = []
    desc_list = []
    valid_idx = []

    for i, smiles in enumerate(df_agg["canonical_smiles"]):
        mol = Chem.MolFromSmiles(smiles)

        # Extract features using the single 'mol' object
        fp = morgan_fp(mol).astype(np.float32)
        desc = np.asarray(physchem(mol), dtype=np.float32)

        # Keep only molecules whose fingerprint AND descriptors are fully valid.
        if not (np.isnan(fp).any() or np.isnan(desc).any()):
            fp_list.append(fp)
            desc_list.append(desc)
            valid_idx.append(i)

    # Convert to arrays
    fp_matrix = np.vstack(fp_list)
    desc_matrix = np.vstack(desc_list)

    df_clean = df_agg.iloc[valid_idx].reset_index(drop=True)
    logging.info("Featurised %d/%d molecules (%d dropped as invalid/NaN)",
                 len(df_clean), len(df_agg), len(df_agg) - len(df_clean))

    # X data split for two stage feature selection
    X_fp   = fp_matrix.astype(np.float32)
    X_desc = desc_matrix.astype(np.float32)
    y      = df_clean["active"].to_numpy()

    FP_COLS       = [f"fp_{i}" for i in range(FP_BITS)]
    FEATURE_NAMES = FP_COLS + DESC_COLS

    logging.info("Final dataset: X_fp = %s, X_desc = %s", X_fp.shape, X_desc.shape)
    logging.info("  FP bits: %d, physchem: %d, EState: %d, fragments: %d",
                 FP_BITS, len(PHYS_FUNCS), len(ESTATE_NAMES), len(FRAG_FUNCS))
    logging.info("  Total descriptors (prefiltered): %d", N_DESC)
    logging.info("Active: %d, Inactive: %d", int(y.sum()), int((y == 0).sum()))

    # Scaffold split + two-stage feature selection
    def _bms_scaffold(mol) -> str:
        """
        Extracts the Bemis-Murcko framework (scaffold) of a given molecule.

        The scaffold consists of the core ring systems and the linker atoms connecting
        them, with all terminal side chains removed. This is primarily used to group
        structurally similar compounds during cross-validation splitting.

        Args:
            mol (rdkit.Chem.rdchem.Mol): The parsed RDKit molecule object.

        Returns:
            str: The non-isomeric SMILES string of the molecular scaffold. Returns an
                 empty string if the scaffold cannot be generated or the molecule is invalid.
        """

        if mol is None:
            return ""
        try:
            sc = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(sc, isomericSmiles=False) if sc else ""
        except Exception:
            return ""

    def scaffold_split(smiles, y, test_frac=0.2, seed=42):
        """
        Splits a dataset into training and testing sets based on Bemis-Murcko scaffolds.

        This ensures that molecules sharing the same core scaffold are kept together
        in either the train or test set. This evaluates the model's ability to
        generalize to novel chemical structures rather than memorizing slight variations.

        Args:
            smiles (list): A list of canonical SMILES strings.
            y (numpy.ndarray): An array of binary target labels.
            test_frac (float, optional): The fraction of data to allocate to the test set.
                                             Defaults to 0.20.
            seed (int, optional): The random seed for reproducibility. Defaults to 42.

        Returns:
            tuple: Two numpy arrays containing the integer indices for the
                   training and testing sets (train_idx, test_idx).
        """
        rng = np.random.RandomState(seed)
        scaffolds = {}
        for i, s in enumerate(smiles):
            # convert SMILES to mol
            try:
                mol = Chem.MolFromSmiles(s)
            except Exception:
                mol = None
            scaffolds.setdefault(_bms_scaffold(mol), []).append(i)
        groups = sorted(scaffolds.values(), key=lambda g: (-len(g), rng.rand()))
        n = len(y)
        n_test = int(round(test_frac * n))
        test_idx, train_idx = [], []
        for g in reversed(groups):
            if len(test_idx) + len(g) <= n_test:
                test_idx.extend(g)
            else:
                train_idx.extend(g)
        used = set(test_idx) | set(train_idx)
        for i in range(n):
            if i not in used:
                train_idx.append(i)
        # Safety - ensure test set is not empty
        if len(test_idx) == 0:
            # move up to n_test samples from train to test
            move_n = min(len(train_idx), max(1, n_test))
            rng.shuffle(train_idx)
            moved = train_idx[:move_n]
            test_idx.extend(moved)
            train_idx = train_idx[move_n:]

        return np.array(sorted(train_idx), dtype=int), np.array(sorted(test_idx), dtype=int)

    def build_feature_pipeline(
        X_fp_tr,
        X_fp_te,
        X_desc_tr,
        X_desc_te,
        y_tr,
        k_fp=950,
    ):
        """
        Applies FP variance filtering, chi2 selection, and scaling.

        Fits on training data only to avoid leakage, then transforms test data.
        Returns scaled features plus FP selections and masks for name tracking.
        """
        vt = VarianceThreshold(threshold=0.0)
        X_fp_tr_vt = vt.fit_transform(X_fp_tr)
        X_fp_te_vt = vt.transform(X_fp_te)
        fp_mask_vt = vt.get_support()

        k_fp_used = min(k_fp, X_fp_tr_vt.shape[1])
        kbs = SelectKBest(chi2, k=k_fp_used)
        X_fp_tr_sel = kbs.fit_transform(X_fp_tr_vt, y_tr)
        X_fp_te_sel = kbs.transform(X_fp_te_vt)
        fp_mask_kbs = kbs.get_support()

        X_tr_raw = np.hstack([X_fp_tr_sel, X_desc_tr]).astype(np.float32)
        X_te_raw = np.hstack([X_fp_te_sel, X_desc_te]).astype(np.float32)

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_raw)
        X_te_scaled = scaler.transform(X_te_raw)

        return (
            X_tr_scaled,
            X_te_scaled,
            X_fp_tr_sel,
            X_fp_te_sel,
            fp_mask_vt,
            fp_mask_kbs,
            k_fp_used,
        )

    smiles_list = df_clean["canonical_smiles"].tolist()

    # main split (seed=42)
    tr_idx, te_idx = scaffold_split(smiles_list, y, test_frac=0.20, seed=RANDOM_STATE)

    X_fp_train,   X_fp_test   = X_fp[tr_idx],   X_fp[te_idx]
    X_desc_train, X_desc_test = X_desc[tr_idx], X_desc[te_idx]
    y_train, y_test = y[tr_idx], y[te_idx]

    (
        X_train_scaled,
        X_test_scaled,
        X_fp_train_sel,
        X_fp_test_sel,
        fp_mask_vt,
        fp_mask_kbs,
        K_FP,
    ) = build_feature_pipeline(
        X_fp_train,
        X_fp_test,
        X_desc_train,
        X_desc_test,
        y_train,
        k_fp=950,
    )

    # Stage B: descriptors
    # Combining fingerprints and descriptors
    X_train = np.hstack([X_fp_train_sel, X_desc_train]).astype(np.float32)
    X_test  = np.hstack([X_fp_test_sel,  X_desc_test]).astype(np.float32)

    # Feature names post selection
    fp_names_after_vt = [n for n, k in zip(FP_COLS, fp_mask_vt) if k]
    fp_kept_names = [n for n, k in zip(fp_names_after_vt, fp_mask_kbs) if k]
    kept_names = fp_kept_names + DESC_COLS

    # Data for kNN-Jaccard
    X_train_bool = X_fp_train_sel.astype(bool)
    X_test_bool  = X_fp_test_sel.astype(bool)

    logging.info("Main split (seed=%d): scaffold-aware (Bemis-Murcko)", RANDOM_STATE)
    logging.info("  Train: FP %s | Desc %s", X_fp_train.shape, X_desc_train.shape)
    logging.info("  Test:  FP %s | Desc %s", X_fp_test.shape, X_desc_test.shape)
    logging.info("  Post VT na FP: %d -> %d", X_fp_train.shape[1], int(np.sum(fp_mask_vt)))
    logging.info("  Post chi squared on FP (k=%d): %d -> %d", K_FP, int(np.sum(fp_mask_vt)), X_fp_train_sel.shape[1])
    logging.info("  Final: X_train = %s, X_test = %s", X_train.shape, X_test.shape)
    logging.info("  Train active/inactive: %d/%d", int(y_train.sum()), int((y_train==0).sum()))
    logging.info("  Test  active/inactive: %d/%d", int(y_test.sum()), int((y_test==0).sum()))


    train_smiles = [smiles_list[i] for i in tr_idx]
    train_scaffold_ids = []
    scaffold_to_id = {}
    for s in train_smiles:
        try:
            mol = Chem.MolFromSmiles(s)
        except Exception:
            mol = None
        sc = _bms_scaffold(mol)
        if sc not in scaffold_to_id:
            scaffold_to_id[sc] = len(scaffold_to_id)
        train_scaffold_ids.append(scaffold_to_id[sc])
    train_scaffold_ids = np.array(train_scaffold_ids, dtype=int)
    logging.info(
        "Train scaffold groups: %d unique scaffolds across %d molecules",
        len(scaffold_to_id), len(train_scaffold_ids),
    )

    cv_groups = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    # all receive an identical, scaffold-disjoint partition.
    cv = list(cv_groups.split(X_train, y_train, groups=train_scaffold_ids))

    # Diagnostic - report fold sizes and class balance
    for i, (tr_f, te_f) in enumerate(cv):
        pos = int(y_train[te_f].sum())
        neg = int((y_train[te_f] == 0).sum())
        logging.info("  CV fold %d: train=%d, val=%d (val active/inactive=%d/%d)",
                     i, len(tr_f), len(te_f), pos, neg)

    def eval_model(name, model, X_tr, y_tr, X_te, y_te):
        """
        Trains a classification model and calculates standard performance metrics.

        Args:
            name (str): The human-readable name of the model.
            model (sklearn.base.BaseEstimator): An uninitialized scikit-learn model.
            X_tr (numpy.ndarray): The training feature matrix.
            y_tr (numpy.ndarray): The training labels.
            X_te (numpy.ndarray): The testing feature matrix.
            y_te (numpy.ndarray): The testing labels.

        Returns:
            dict: A dictionary containing the model name, balanced accuracy,
                  accuracy, AUC-ROC, MCC, raw predictions, and raw probabilities.
        """

        model.fit(X_tr, y_tr)
        y_pred  = model.predict(X_te)
        y_proba = (model.predict_proba(X_te)[:, 1]
                   if hasattr(model, "predict_proba") else model.decision_function(X_te))
        return {
            "model":   name, "bal_acc": balanced_accuracy_score(y_te, y_pred),
            "accuracy": accuracy_score(y_te, y_pred),
            "auc_roc": safe_roc_auc(y_te, y_proba),
            "mcc":     matthews_corrcoef(y_te, y_pred),
            "_pred":   y_pred, "_proba": y_proba, "_obj": model,
        }

    baseline = {}
    baseline["LogReg"] = eval_model(
        "LogReg-L2",
        LogisticRegression(max_iter=4000, class_weight="balanced",
                           solver="liblinear", C=1.0, random_state=RANDOM_STATE),
        X_train_scaled, y_train, X_test_scaled, y_test,
    )
    baseline["kNN"] = eval_model(
        "kNN-Jaccard",
        KNeighborsClassifier(n_neighbors=7, weights="distance", metric="jaccard"),
        X_train_bool, y_train, X_test_bool, y_test,
    )
    baseline["RF"] = eval_model(
        "RF (baseline)",
        RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                               n_jobs=-1, random_state=RANDOM_STATE),
        X_train, y_train, X_test, y_test,
    )
    baseline["ET"] = eval_model(
        "ExtraTrees (baseline)",
        ExtraTreesClassifier(n_estimators=400, class_weight="balanced_subsample",
                             n_jobs=-1, random_state=RANDOM_STATE),
        X_train, y_train, X_test, y_test,
    )
    baseline["HGB"] = eval_model(
        "HistGB (baseline)",
        HistGradientBoostingClassifier(class_weight="balanced",
                                       early_stopping=True, validation_fraction=0.15,
                                       random_state=RANDOM_STATE),
        X_train, y_train, X_test, y_test,
    )
    baseline["SVM"] = eval_model(
        "SVM-RBF",
        SVC(kernel="rbf", C=2.0, gamma="scale",
            class_weight="balanced", probability=True, random_state=RANDOM_STATE),
        X_train_scaled, y_train, X_test_scaled, y_test,
    )
    baseline["LGBM"] = eval_model(
        "LightGBM",
        lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.05, num_leaves=63,
            min_child_samples=20, subsample=0.9, subsample_freq=1,
            colsample_bytree=0.8, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        ),
        X_train, y_train, X_test, y_test,
    )

    print(f"{'Model':<24} {'bal_acc':>8} {'accuracy':>9} {'auc_roc':>8} {'MCC':>6}")
    print("-" * 58)
    for r in baseline.values():
        print(f"{r['model']:<24} {r['bal_acc']:>8.3f} {r['accuracy']:>9.3f} "
              f"{r['auc_roc']:>8.3f} {r['mcc']:>6.3f}")

    # HalvingRandomSearchCV for Random Forest Classifier
    rf_grid = {
        "n_estimators":      [300, 500, 800, 1200],
        "max_depth":         [None, 15, 25, 40],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "max_features":      ["sqrt", 0.1, 0.2],
        "class_weight":      ["balanced", "balanced_subsample"],
    }
    rf_search = HalvingRandomSearchCV(
        RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE),
        rf_grid, cv=cv, scoring="balanced_accuracy",
        factor=3, resource="n_samples",
        min_resources=100, max_resources=len(y_train),
        n_candidates=60, random_state=RANDOM_STATE,
        n_jobs=-1, verbose=1, refit=True,
    )
    rf_search.fit(X_train, y_train)
    best_rf  = rf_search.best_estimator_
    rf_pred  = best_rf.predict(X_test)
    rf_proba = best_rf.predict_proba(X_test)[:, 1]
    logging.info("\nBest RF: %s", rf_search.best_params_)
    logging.info("CV bal_acc: %.4f | Test bal_acc: %.4f | MCC: %.4f | AUC: %.4f",
                 rf_search.best_score_, balanced_accuracy_score(y_test, rf_pred),
                 matthews_corrcoef(y_test, rf_pred), safe_roc_auc(y_test, rf_proba))

    # HalvingRandomSearchCV for HistGradientBoosting
    N_NEG, N_POS = int((y_train == 0).sum()), int((y_train == 1).sum())
    balanced_ratio = N_POS / N_NEG

    # Subsidiary weight modification- no pos_weight option
    class_weights_options = [
        {0: 1.0, 1: 1.0},
        {0: 3.0, 1: 1.0},
        "balanced",
        {0: 7.0, 1: 1.0},
    ]

    hgb_grid = {
        "learning_rate":     [0.02, 0.03, 0.05, 0.08, 0.1],
        "max_iter":          [400, 700, 1000],
        "max_leaf_nodes":    [15, 31, 63, 127],
        "min_samples_leaf":  [5, 10, 20, 40],
        "l2_regularization": [0.0, 0.5, 1.0, 2.0],
        "max_features":      [0.8, 1.0],
        "class_weight":      class_weights_options,
    }
    hgb_search = HalvingRandomSearchCV(
        HistGradientBoostingClassifier(
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=25,
            random_state=RANDOM_STATE,
        ),
        hgb_grid, cv=cv, scoring="balanced_accuracy",
        factor=3, resource="n_samples",
        min_resources=100, max_resources=len(y_train),
        n_candidates=80, random_state=RANDOM_STATE,
        n_jobs=-1, verbose=1, refit=True,
    )
    hgb_search.fit(X_train, y_train)
    best_hgb  = hgb_search.best_estimator_
    hgb_pred  = best_hgb.predict(X_test)
    hgb_proba = best_hgb.predict_proba(X_test)[:, 1]
    logging.info("\nBest HGB: %s", hgb_search.best_params_)
    logging.info("CV bal_acc: %.4f | Test bal_acc: %.4f | MCC: %.4f | AUC: %.4f",
                 hgb_search.best_score_, balanced_accuracy_score(y_test, hgb_pred),
                 matthews_corrcoef(y_test, hgb_pred), safe_roc_auc(y_test, hgb_proba))

    # HalvingRandomSearchCV for LightGBM with scale_pos_weight grid
    lgbm_grid = {
        "n_estimators":       [400, 700, 1000, 1500],
        "learning_rate":      [0.02, 0.03, 0.05, 0.08, 0.1],
        "num_leaves":         [31, 63, 127, 255],
        "min_child_samples":  [5, 10, 20, 40],
        "subsample":          [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree":   [0.6, 0.8, 1.0],
        "reg_lambda":         [0.0, 0.5, 1.0, 2.0],
        "scale_pos_weight":   [0.14, 0.20, 0.33, 1.0],
    }
    lgbm_search = HalvingRandomSearchCV(
        lgb.LGBMClassifier(
            subsample_freq=1, random_state=RANDOM_STATE,
            n_jobs=-1, verbose=-1,
        ),
        lgbm_grid, cv=cv, scoring="balanced_accuracy",
        factor=3, resource="n_samples",
        min_resources=100, max_resources=len(y_train),
        n_candidates=80, random_state=RANDOM_STATE,
        n_jobs=-1, verbose=1, refit=True,
    )
    lgbm_search.fit(X_train, y_train)
    best_lgbm  = lgbm_search.best_estimator_
    lgbm_pred  = best_lgbm.predict(X_test)
    lgbm_proba = best_lgbm.predict_proba(X_test)[:, 1]
    logging.info("\nBest LightGBM: %s", lgbm_search.best_params_)
    logging.info("CV bal_acc: %.4f | Test bal_acc: %.4f | MCC: %.4f | AUC: %.4f",
                 lgbm_search.best_score_, balanced_accuracy_score(y_test, lgbm_pred),
                 matthews_corrcoef(y_test, lgbm_pred), safe_roc_auc(y_test, lgbm_proba))

    # HalvingRandomSearchCV for ExtraTrees
    et_grid = {
        "n_estimators":      [300, 500, 800, 1200],
        "max_depth":         [None, 15, 25, 40],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "max_features":      ["sqrt", 0.1, 0.2],
        "class_weight":      ["balanced", "balanced_subsample"],
    }
    et_search = HalvingRandomSearchCV(
        ExtraTreesClassifier(n_jobs=-1, random_state=RANDOM_STATE),
        et_grid, cv=cv, scoring="balanced_accuracy",
        factor=3, resource="n_samples",
        min_resources=100, max_resources=len(y_train),
        n_candidates=60, random_state=RANDOM_STATE,
        n_jobs=-1, verbose=1, refit=True,
    )
    et_search.fit(X_train, y_train)
    best_et  = et_search.best_estimator_
    et_pred  = best_et.predict(X_test)
    et_proba = best_et.predict_proba(X_test)[:, 1]
    logging.info("\nBest ExtraTrees: %s", et_search.best_params_)
    logging.info("CV bal_acc: %.4f | Test bal_acc: %.4f | MCC: %.4f | AUC: %.4f",
                 et_search.best_score_, balanced_accuracy_score(y_test, et_pred),
                 matthews_corrcoef(y_test, et_pred), safe_roc_auc(y_test, et_proba))

    # Isotonic calibration and dual threshold
    def find_thresholds(model, X_tr, y_tr, cv_):
        """
        Determines optimal probability thresholds for screening and decision making.

        Uses out-of-fold predictions to find the threshold that maximizes the F2 score
        (favoring recall for initial screening) and the F0.5 score (favoring precision
        for final decisions).

        Args:
            model (sklearn.base.BaseEstimator): The fitted estimator/pipeline.
            X_tr (numpy.ndarray): The training feature matrix.
            y_tr (numpy.ndarray): The training labels.
            cv_ (sklearn.model_selection.BaseCrossValidator): The cross-validation strategy.

        Returns:
            tuple: A tuple containing (t_screen, t_decision, proba_oof), where t_screen
                   and t_decision are calculated threshold floats, and proba_oof is a
                   1D numpy array of out-of-fold probabilities.
        """
        proba_oof = cross_val_predict(model, X_tr, y_tr, cv=cv_,
                                      method="predict_proba", n_jobs=-1)[:, 1]
        thresholds = np.linspace(0.05, 0.95, 181)
        f2_neg  = []   # F2 negative (high recall)
        f05_neg = []   # F0.5 negative (high precision)
        for t in thresholds:
            pred = (proba_oof >= t).astype(int)
            f2_neg.append(fbeta_score(y_tr, pred, beta=2.0, pos_label=0, zero_division=0))
            f05_neg.append(fbeta_score(y_tr, pred, beta=0.5, pos_label=0, zero_division=0))
        t_screen   = float(thresholds[int(np.argmax(f2_neg))])
        t_decision = float(thresholds[int(np.argmax(f05_neg))])
        return t_screen, t_decision, proba_oof

    # Best model - CV balanced_accuracy
    all_candidates = {
        "RF*":   (best_rf,   rf_search.best_score_),
        "HGB*":  (best_hgb,  hgb_search.best_score_),
        "LGBM*": (best_lgbm, lgbm_search.best_score_),
        "ET*":   (best_et,   et_search.best_score_),
    }
    best_name  = max(all_candidates, key=lambda k: all_candidates[k][1])
    best_model = all_candidates[best_name][0]
    logging.info("Chosen for calibration: %s (CV bal_acc = %.4f)", best_name, all_candidates[best_name][1])

    calibrated = CalibratedClassifierCV(best_model, method="isotonic", cv=3)
    calibrated.fit(X_train, y_train)

    t_screen, t_decision, _ = find_thresholds(calibrated, X_train, y_train, cv)
    proba_final  = calibrated.predict_proba(X_test)[:, 1]

    pred_default  = (proba_final >= 0.5).astype(int)
    pred_screen   = (proba_final >= t_screen).astype(int)
    pred_decision = (proba_final >= t_decision).astype(int)

    def report_threshold(label, t, pred):
        logging.info("  %s (t=%.2f): bal_acc=%.4f | MCC=%.4f | Prec(0)=%.3f | Rec(0)=%.3f",
                     label, t, balanced_accuracy_score(y_test, pred),
                     matthews_corrcoef(y_test, pred),
                     precision_score(y_test, pred, pos_label=0, zero_division=0),
                     recall_score(y_test, pred, pos_label=0, zero_division=0))

    logging.info("\nDual threshold (%s + isotonic):", best_name)
    report_threshold("@default",  0.5,        pred_default)
    report_threshold("@screen",   t_screen,   pred_screen)
    report_threshold("@decision", t_decision, pred_decision)
    logging.info("\nTest AUC-ROC (calibrated): %.4f", safe_roc_auc(y_test, proba_final))

    # Calibrated stacking: isotonic CalibratedClassifierCV
    def make_calibrated(base):
        return CalibratedClassifierCV(clone(base), method="isotonic", cv=3)

    # Base specs
    base_specs = [
        ("rf",   make_calibrated(best_rf),   X_train,      X_test),
        ("et",   make_calibrated(best_et),   X_train,      X_test),
        ("hgb",  make_calibrated(best_hgb),  X_train,      X_test),
        ("lgbm", make_calibrated(best_lgbm), X_train,      X_test),
        ("knn",  make_calibrated(KNeighborsClassifier(
                                     n_neighbors=7, weights="distance", metric="jaccard")),
                                              X_train_bool, X_test_bool),
    ]

    oof_train = np.zeros((len(y_train), len(base_specs)))
    oof_test  = np.zeros((len(y_test),  len(base_specs)))

    logging.info("Training calibrated base models")
    for j, (name, mdl, Xtr, Xte) in enumerate(base_specs):
        oof_train[:, j] = cross_val_predict(
            mdl, Xtr, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        mdl.fit(Xtr, y_train)
        oof_test[:, j]  = mdl.predict_proba(Xte)[:, 1]
        bacc = balanced_accuracy_score(y_train, (oof_train[:, j] >= 0.5).astype(int))
        mcc  = matthews_corrcoef(y_train, (oof_train[:, j] >= 0.5).astype(int))
        logging.info("  [%s] OOF bal_acc=%.4f | OOF MCC=%.4f", name, bacc, mcc)

    meta = LogisticRegression(C=1.0, class_weight="balanced",
                              max_iter=2000, random_state=RANDOM_STATE)
    meta.fit(oof_train, y_train)

    stack_proba = meta.predict_proba(oof_test)[:, 1]
    stack_pred  = (stack_proba >= 0.5).astype(int)

    logging.info("\nMeta-learner learned weights:")
    for j, (name, _, _, _) in enumerate(base_specs):
        logging.info("  %s: coef=%+.3f", name, meta.coef_[0, j])
    logging.info("  intercept: %+.3f", meta.intercept_[0])

    logging.info("\nCalibrated Stacking – test:")
    logging.info("  bal_acc: %.4f", balanced_accuracy_score(y_test, stack_pred))
    logging.info("  MCC    : %.4f", matthews_corrcoef(y_test, stack_pred))
    logging.info("  AUC-ROC: %.4f", safe_roc_auc(y_test, stack_proba))
    logging.info("  Prec(0): %.3f | Rec(0): %.3f",
                 precision_score(y_test, stack_pred, pos_label=0, zero_division=0),
                 recall_score(y_test, stack_pred, pos_label=0, zero_division=0))

    # Soft Voting
    oof_mcc = np.zeros(len(base_specs))
    for j in range(len(base_specs)):
        oof_mcc[j] = matthews_corrcoef(y_train, (oof_train[:, j] >= 0.5).astype(int))

    # Softmax weights - MCC
    weights = np.exp(oof_mcc * 5)   # temperature = 5
    weights /= weights.sum()

    logging.info("Soft voting weights (softmax MCC, T=5):")
    for j, (name, *_ ) in enumerate(base_specs):
        logging.info("  %s: MCC=%.4f -> w=%.3f", name, oof_mcc[j], weights[j])

    vote_proba = oof_test @ weights
    vote_pred  = (vote_proba >= 0.5).astype(int)

    logging.info("\nSoft Voting – test:")
    logging.info("  bal_acc: %.4f", balanced_accuracy_score(y_test, vote_pred))
    logging.info("  MCC    : %.4f", matthews_corrcoef(y_test, vote_pred))
    logging.info("  AUC-ROC: %.4f", safe_roc_auc(y_test, vote_proba))
    logging.info("  Prec(0): %.3f | Rec(0): %.3f",
                 precision_score(y_test, vote_pred, pos_label=0, zero_division=0),
                 recall_score(y_test, vote_pred, pos_label=0, zero_division=0))

    # Multi-seed scaffold CV - top 5 models
    SEEDS = [42, 17, 99, 256, 1337]

    def build_pipeline_for_seed(seed):
        """
        Constructs a complete data processing pipeline for a specific random seed.

        This isolates the data preparation for multi-seed validation. It performs
        scaffold-aware splitting, two-stage feature selection (VarianceThreshold
        followed by SelectKBest via chi-squared), and independent feature scaling
        to strictly prevent data leakage between folds.

        Args:
            seed (int): The random seed used for generating the data split.

        Returns:
            tuple: Contains scaled training features (Xtr), scaled testing features (Xte),
                   boolean masks for binary features (Xtr_b, Xte_b), training labels (yt),
                   and validation labels (yv).
        """

        tr, te = scaffold_split(smiles_list, y, test_frac=0.20, seed=seed)
        Xf_tr, Xf_te = X_fp[tr], X_fp[te]
        Xd_tr, Xd_te = X_desc[tr], X_desc[te]
        yt, yv = y[tr], y[te]

        Xtr_scaled, Xte_scaled, Xf_tr_s, Xf_te_s, _, _, _ = build_feature_pipeline(
            Xf_tr,
            Xf_te,
            Xd_tr,
            Xd_te,
            yt,
            k_fp=950,
        )

        return Xtr_scaled, Xte_scaled, Xf_tr_s.astype(bool), Xf_te_s.astype(bool), yt, yv
    def make_top_models():
        return {
            "RF_HalvingRS":   clone(best_rf),
            "HGB_HalvingRS":  clone(best_hgb),
            "LGBM_HalvingRS": clone(best_lgbm),
            "ET_HalvingRS":   clone(best_et),
            "kNN-Jaccard":    KNeighborsClassifier(n_neighbors=7, weights="distance", metric="jaccard"),
        }

    multiseed_results = {name: {"bal_acc": [], "mcc": [], "auc": []}
                        for name in make_top_models()}

    logging.info("Multi-seed scaffold CV (%d seeds)…", len(SEEDS))
    for seed in SEEDS:
        logging.info("  seed=%d…", seed)
        Xtr, Xte, Xtr_b, Xte_b, yt, yv = build_pipeline_for_seed(seed)
        for name, mdl in make_top_models().items():
            if name == "kNN-Jaccard":
                mdl.fit(Xtr_b, yt)
                pred  = mdl.predict(Xte_b)
                proba = mdl.predict_proba(Xte_b)[:, 1]
            else:
                mdl.fit(Xtr, yt)
                pred  = mdl.predict(Xte)
                proba = mdl.predict_proba(Xte)[:, 1]
            multiseed_results[name]["bal_acc"].append(balanced_accuracy_score(yv, pred))
            multiseed_results[name]["mcc"].append(matthews_corrcoef(yv, pred))
            multiseed_results[name]["auc"].append(safe_roc_auc(yv, proba))

    print(f"\n{'Model':<18} {'bal_acc':>15} {'MCC':>15} {'AUC':>15}")
    print("-" * 65)
    for name, vals in multiseed_results.items():
        ba = np.array(vals["bal_acc"])
        mc = np.array(vals["mcc"])
        au = np.array(vals["auc"])
        print(f"{name:<18} {ba.mean():.3f} ± {ba.std():.3f}   "
              f"{mc.mean():.3f} ± {mc.std():.3f}   "
              f"{au.mean():.3f} ± {au.std():.3f}")

    # Results
    final_rows = []
    for r in baseline.values():
        final_rows.append({
            "Model":             r["model"],
            "balanced_accuracy": r["bal_acc"], "MCC": r["mcc"],
            "accuracy": r["accuracy"], "AUC-ROC": r["auc_roc"],
        })
    for name, pred, proba in [
        ("RF (HalvingRS)",       rf_pred,    rf_proba),
        ("HistGB (HalvingRS)",   hgb_pred,   hgb_proba),
        ("LightGBM (HalvingRS)", lgbm_pred,  lgbm_proba),
        ("ExtraTrees (HalvingRS)", et_pred,  et_proba),
        (f"{best_name} + isotonic @0.50",       pred_default,  proba_final),
        (f"{best_name} + isotonic @{t_screen:.2f} (screen)",   pred_screen,   proba_final),
        (f"{best_name} + isotonic @{t_decision:.2f} (decision)", pred_decision, proba_final),
        ("Calibrated STACK",     stack_pred, stack_proba),
        ("Soft Voting",          vote_pred,  vote_proba),
    ]:
        final_rows.append({
            "Model":             name,
            "balanced_accuracy": balanced_accuracy_score(y_test, pred),
            "MCC":               matthews_corrcoef(y_test, pred),
            "accuracy":          accuracy_score(y_test, pred),
            "AUC-ROC":           safe_roc_auc(y_test, proba),
        })

    results_df = pd.DataFrame(final_rows)
    print("=== balanced_accuracy ranking ===")
    print(results_df.sort_values("balanced_accuracy", ascending=False)
          .reset_index(drop=True)
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== MCC ranking ===")
    print(results_df.sort_values("MCC", ascending=False)
          .reset_index(drop=True)
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    best_bal = results_df.sort_values("balanced_accuracy", ascending=False).iloc[0]
    best_mcc = results_df.sort_values("MCC", ascending=False).iloc[0]
    print(f"\n*** Best bal_acc: {best_bal['balanced_accuracy']:.4f} ({best_bal['Model']}) ***")
    print(f"*** Best MCC    : {best_mcc['MCC']:.4f} ({best_mcc['Model']}) ***")

    test_smiles = df_clean.loc[te_idx, "canonical_smiles"].values
    test_chembl = (
        df_clean.loc[te_idx, "molecule_chembl_id"].values
        if "molecule_chembl_id" in df_clean.columns
        else [""] * len(te_idx)
    )
    preds_df = pd.DataFrame({
        "molecule_chembl_id": test_chembl,
        "smiles":             test_smiles,
        "true_label":         y_test,
        "proba_rf":           rf_proba,
        "proba_hgb":          hgb_proba,
        "proba_lgbm":         lgbm_proba,
        "proba_et":           et_proba,
        "proba_calib":        proba_final,
        "proba_stack":        stack_proba,
        "proba_vote":         vote_proba,
    })
    preds_df["error_stack"] = (stack_pred != y_test).astype(int)
    out_csv = "test_predictions_v7.csv"
    preds_df.to_csv(out_csv, index=False)
    logging.info("Export: %s (%d lines)", out_csv, len(preds_df))

    # ROC curve and permutation feature importance
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    ax = axes[0]
    roc_curves = [
        ("LogReg",          baseline["LogReg"]["_proba"]),
        ("kNN-Jaccard",     baseline["kNN"]["_proba"]),
        ("RF baseline",     baseline["RF"]["_proba"]),
        ("ExtraTrees base", baseline["ET"]["_proba"]),
        ("HistGB baseline", baseline["HGB"]["_proba"]),
        ("SVM-RBF",         baseline["SVM"]["_proba"]),
        ("LightGBM",        baseline["LGBM"]["_proba"]),
        ("RF HalvingRS",    rf_proba),
        ("HGB HalvingRS",   hgb_proba),
        ("LGBM HalvingRS",  lgbm_proba),
        ("ET HalvingRS",    et_proba),
        (f"{best_name} calib", proba_final),
        ("Calib STACK",     stack_proba),
        ("Soft Voting",     vote_proba),
    ]
    for name, proba in roc_curves:
        try:
            fpr, tpr, _ = roc_curve(y_test, proba)
            auc_val = safe_roc_auc(y_test, proba)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})", lw=1.1)
        except Exception:
            logging.warning("Skipping ROC curve for %s (could not compute ROC/AUC)", name)
            continue
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves")
    ax.legend(loc="lower right", fontsize=7)

    ax = axes[1]
    perm = permutation_importance(
        best_model, X_test, y_test, n_repeats=10,
        scoring="balanced_accuracy", random_state=RANDOM_STATE, n_jobs=-1,
    )
    importances = perm.importances_mean
    top_n = 15
    top_idx = np.argsort(importances)[::-1][:top_n]
    top_names = [kept_names[i] for i in top_idx]
    top_vals  = importances[top_idx]
    top_err   = perm.importances_std[top_idx]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
    ax.barh(range(top_n), top_vals, xerr=top_err, color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names)
    ax.invert_yaxis()
    ax.set_xlabel("Balanced_accuracy (permutation)")
    ax.set_title(f"Top {top_n} features ({best_name})")
    plt.tight_layout()
    plt.savefig("figure_1_v7.svg", format="svg", bbox_inches="tight")
    plt.close()

    logging.info("Molecular descriptors – permutation importance:")
    for desc in DESC_COLS:
        if desc in kept_names:
            idx = kept_names.index(desc)
            logging.info("  %s: %+.4f  (± %+.4f)", desc, importances[idx], perm.importances_std[idx])

    # Confusion matrix for top 3 models
    models_for_eval = [
        ("LogReg-L2",            baseline["LogReg"]["_pred"]),
        ("kNN-Jaccard",          baseline["kNN"]["_pred"]),
        ("RF baseline",          baseline["RF"]["_pred"]),
        ("ExtraTrees baseline",  baseline["ET"]["_pred"]),
        ("HistGB baseline",      baseline["HGB"]["_pred"]),
        ("SVM-RBF",              baseline["SVM"]["_pred"]),
        ("LightGBM",             baseline["LGBM"]["_pred"]),
        ("RF HalvingRS",         rf_pred),
        ("HGB HalvingRS",        hgb_pred),
        ("LGBM HalvingRS",       lgbm_pred),
        ("ET HalvingRS",         et_pred),
        (f"{best_name} @0.5",    pred_default),
        (f"{best_name} screen",  pred_screen),
        (f"{best_name} decision", pred_decision),
        ("Calib STACK",          stack_pred),
        ("Soft Voting",          vote_pred),
    ]

    ext_rows = []
    for name, y_pred in models_for_eval:
        ext_rows.append({
            "Model":   name,
            "bal_acc": balanced_accuracy_score(y_test, y_pred),
            "MCC":     matthews_corrcoef(y_test, y_pred),
            "F1(1)":   f1_score(y_test, y_pred, pos_label=1, zero_division=0),
            "F1(0)":   f1_score(y_test, y_pred, pos_label=0, zero_division=0),
            "Prec(0)": precision_score(y_test, y_pred, pos_label=0, zero_division=0),
            "Rec(0)":  recall_score(y_test, y_pred, pos_label=0, zero_division=0),
        })
    ext_df = (pd.DataFrame(ext_rows).sort_values("MCC", ascending=False)
              .reset_index(drop=True))
    print(ext_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # Heatmaps for top 3 models - MCC
    top3 = ext_df.head(3)["Model"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, name in zip(axes, top3):
        pred = dict(models_for_eval)[name]
        cm = confusion_matrix(y_test, pred)
        im = ax.imshow(cm, cmap="Blues", aspect="auto")

        ax.grid(False)

        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["True 0", "True 1"])
        ax.set_title(f"{name}\nMCC={matthews_corrcoef(y_test, pred):.3f}")
    plt.tight_layout()
    plt.savefig("figure_2_v7.svg", format="svg", bbox_inches="tight")
    plt.close()

    # Bar charts MCC a F1(0)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, col, title in zip(axes, ["MCC", "F1(0)"],
                              ["Matthews correlation coefficient",
                               "F1 – inactive class (0)"]):
        ax.bar(ext_df["Model"], ext_df[col],
               color=plt.cm.viridis(np.linspace(0.2, 0.9, len(ext_df))))
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(ext_df[col]):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold", fontsize=8)
    plt.tight_layout()
    plt.savefig("figure_3_v7.svg", format="svg", bbox_inches="tight")
    plt.close()

    # Final report
    best_row  = ext_df.iloc[0]["Model"]
    best_pred = dict(models_for_eval)[best_row]
    print(f"\nClassification report – highest MCC model: {best_row}")
    print(classification_report(y_test, best_pred,
                                target_names=["Inactive (0)", "Active (1)"]))


if __name__ == "__main__":
    main()
