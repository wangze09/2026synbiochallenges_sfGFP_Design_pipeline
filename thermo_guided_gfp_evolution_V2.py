import os
import random
import joblib
import numpy as np
import pandas as pd


# ============================================================
# 1. 用户需要修改的参数
# ============================================================

MODEL_FILE = "xgb_gfp_brightness_model.joblib"

# ThermoMPNN 单点突变 ddG 文件
# 需要包含列：Mutation, ddG (kcal/mol), pos, wtAA, mutAA
THERMOMPNN_DDG_FILE = "mutation_ddg_thermoMPNN.csv"

# 你说明 ThermoMPNN 文件里的 pos 需要 +2 才是实际 GFP 位点
THERMOMPNN_POSITION_SHIFT = 2

# ============================================================
# 比赛评分相关参数
# ============================================================

# 如果初始亮度低于 sfGFP WT 的 30%，比赛规则中直接记 0 分
LOW_BRIGHTNESS_RELATIVE_THRESHOLD = 0.30

# 选择分数中，比赛代理分数的权重
COMPETITION_PROXY_WEIGHT = 0.85

# 选择分数中，rank 辅助项的权重
RANK_TIEBREAKER_WEIGHT = 0.15

# rank 辅助项里，亮度和热稳定性的比例
AUX_BRIGHTNESS_RANK_WEIGHT = 0.70
AUX_THERMO_RANK_WEIGHT = 0.30

# 用 ThermoMPNN ddG_sum 估计热处理后荧光保留率：
# estimated_heat_retention = exp(-max(ddG_sum, 0) / HEAT_RETENTION_DDG_SCALE)
# 这个值越小，对正 ddG 去稳定化突变惩罚越强
HEAT_RETENTION_DDG_SCALE = 1.0

# 估计热保留率下限，避免数值变成 0
HEAT_RETENTION_FLOOR = 0.02

# 对 ThermoMPNN 文件中没有覆盖的新突变给一点惩罚
THERMO_UNSCORED_MUTATION_PENALTY = 0.15


# ============================================================
# 进化搜索相关参数
# ============================================================

# 初始全局设置
THERMO_SELECTION_WEIGHT = 0.2
THERMO_GUIDED_MUTATION_PROB = 0.5
THERMO_GUIDED_MAX_DDG = 0.0
THERMO_SAMPLING_TEMPERATURE = 0.5
THERMO_PRESEED_SINGLE_MUTANTS = 100

# 早期探索阶段比例
EARLY_EXPLORATION_FRAC = 0.30

# 后期收敛阶段比例
LATE_EXPLOIT_FRAC = 0.30

# 这里放 avGFP WT 序列，也就是训练模型时默认的 reference
AVGFP_WT_SEQUENCE = """
MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK
"""

# 这里放 sfGFP 序列，作为进化算法的起始序列
SFGFP_START_SEQUENCE = """
MSKGEELFTGVVPILVELDGDVNGHKFSVRGEGEGDATNGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTISFKDDGTYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNFNSHNVYITADKQKNGIKANFKIRHNVEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSVLSKDPNEKRDHMVLLEFVTAAGITHGMDELYK
"""

# 不允许继续突变的位置，1-based 编号
PROTECTED_POSITIONS = {
    1, 30, 39, 65, 80, 99, 105, 145, 153, 163, 171, 206,
}

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

POPULATION_SIZE = 300
N_GENERATIONS = 80

# 相对于 sfGFP，最多允许新增多少个突变
# MAX_NEW_MUTATIONS_VS_SFGFP = 8
MAX_NEW_MUTATIONS_VS_SFGFP = 6


MIN_MUTATIONS_PER_STEP = 1
MAX_MUTATIONS_PER_STEP = 3

ELITE_FRAC = 0.15
PARENT_POOL_FRAC = 0.40

TOP_N = 20

RANDOM_SEED = 42


# ============================================================
# 2. 基础工具函数
# ============================================================

def clean_sequence(seq):
    seq = str(seq)
    seq = seq.replace(" ", "")
    seq = seq.replace("\n", "")
    seq = seq.replace("\r", "")
    seq = seq.upper()
    return seq


def validate_sequence(seq, expected_length, name="sequence"):
    if len(seq) != expected_length:
        raise ValueError(
            f"{name} length is {len(seq)}, but expected length is {expected_length}"
        )

    for i, aa in enumerate(seq, start=1):
        if aa not in AA_LIST:
            raise ValueError(
                f"Invalid amino acid '{aa}' in {name} at position {i}"
            )


def sequence_to_mutation_string(reference_seq, query_seq):
    mutations = []

    for i, (ref_aa, query_aa) in enumerate(
        zip(reference_seq, query_seq),
        start=1
    ):
        if ref_aa != query_aa:
            mutations.append(f"{ref_aa}{i}{query_aa}")

    if len(mutations) == 0:
        return "WT"

    return ":".join(mutations)


def count_mutations(reference_seq, query_seq):
    return sum(
        1 for ref_aa, query_aa in zip(reference_seq, query_seq)
        if ref_aa != query_aa
    )


def get_mutated_positions(reference_seq, query_seq):
    return {
        i
        for i, (ref_aa, query_aa) in enumerate(
            zip(reference_seq, query_seq),
            start=1
        )
        if ref_aa != query_aa
    }


def mutation_label(wt_aa, pos, mut_aa):
    return f"{wt_aa}{int(pos)}{mut_aa}"


def force_protected_positions_to_start_seq(candidate_seq, start_seq, protected_positions):
    candidate_list = list(candidate_seq)

    for pos in protected_positions:
        candidate_list[pos - 1] = start_seq[pos - 1]

    return "".join(candidate_list)


def enforce_max_new_mutations_vs_start(
    candidate_seq,
    start_seq,
    max_new_mutations_vs_start,
    ddg_lookup=None
):
    mutated_positions = list(get_mutated_positions(start_seq, candidate_seq))

    if len(mutated_positions) <= max_new_mutations_vs_start:
        return candidate_seq

    candidate_list = list(candidate_seq)
    n_to_revert = len(mutated_positions) - max_new_mutations_vs_start

    if ddg_lookup is None or len(ddg_lookup) == 0:
        positions_to_revert = random.sample(mutated_positions, n_to_revert)
    else:
        scored_positions = []

        for pos in mutated_positions:
            start_aa = start_seq[pos - 1]
            mut_aa = candidate_seq[pos - 1]

            ddg = ddg_lookup.get((pos, start_aa, mut_aa), np.inf)

            # ddG 越大越差，越应该被回退
            # 没有 ThermoMPNN 覆盖的突变 ddG = inf，也会优先回退
            scored_positions.append((ddg, random.random(), pos))

        scored_positions.sort(reverse=True)
        positions_to_revert = [
            pos for _, _, pos in scored_positions[:n_to_revert]
        ]

    for pos in positions_to_revert:
        candidate_list[pos - 1] = start_seq[pos - 1]

    return "".join(candidate_list)


# ============================================================
# 3. ThermoMPNN ddG 读取与打分
# ============================================================

def find_column_case_insensitive(df, candidate_names):
    normalized = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for name in candidate_names:
        key = name.strip().lower()
        if key in normalized:
            return normalized[key]

    return None


def load_thermompn_ddg_table(
    ddg_file,
    protein_length,
    protected_positions,
    position_shift=2,
    start_seq=None
):
    if ddg_file is None or not os.path.exists(ddg_file):
        print(f"[WARN] ThermoMPNN ddG file not found: {ddg_file}")
        print("[WARN] Search will fall back to brightness-only mode.")
        return None, {}

    raw = pd.read_csv(ddg_file)

    col_ddg = find_column_case_insensitive(
        raw,
        ["ddG (kcal/mol)", "ddG", "ddg", "deltaG", "delta_g"]
    )
    col_pos = find_column_case_insensitive(
        raw,
        ["pos", "position"]
    )
    col_wt = find_column_case_insensitive(
        raw,
        ["wtAA", "wt", "wildtype", "wild_type"]
    )
    col_mut = find_column_case_insensitive(
        raw,
        ["mutAA", "mut", "mutant", "mutation_to"]
    )
    col_mutation = find_column_case_insensitive(
        raw,
        ["Mutation", "mutation"]
    )

    missing = []
    if col_ddg is None:
        missing.append("ddG (kcal/mol)")
    if col_pos is None:
        missing.append("pos")
    if col_wt is None:
        missing.append("wtAA")
    if col_mut is None:
        missing.append("mutAA")

    if missing:
        raise ValueError(
            "ThermoMPNN CSV is missing required columns: "
            + ", ".join(missing)
        )

    ddg_df = pd.DataFrame({
        "original_pos": pd.to_numeric(raw[col_pos], errors="coerce"),
        "wtAA": raw[col_wt].astype(str).str.strip().str.upper(),
        "mutAA": raw[col_mut].astype(str).str.strip().str.upper(),
        "ddG": pd.to_numeric(raw[col_ddg], errors="coerce"),
    })

    if col_mutation is not None:
        ddg_df["original_mutation"] = raw[col_mutation].astype(str)
    else:
        ddg_df["original_mutation"] = None

    ddg_df = ddg_df.dropna(subset=["original_pos", "ddG"]).copy()
    ddg_df["original_pos"] = ddg_df["original_pos"].astype(int)

    # ThermoMPNN 文件中的 pos 需要 +2
    ddg_df["corrected_pos"] = ddg_df["original_pos"] + int(position_shift)

    ddg_df["corrected_mutation"] = [
        mutation_label(wt, pos, mut)
        for wt, pos, mut in zip(
            ddg_df["wtAA"],
            ddg_df["corrected_pos"],
            ddg_df["mutAA"]
        )
    ]

    valid_aa = set(AA_LIST)

    before = len(ddg_df)

    ddg_df = ddg_df[
        ddg_df["corrected_pos"].between(1, protein_length)
        & ddg_df["wtAA"].isin(valid_aa)
        & ddg_df["mutAA"].isin(valid_aa)
        & (ddg_df["wtAA"] != ddg_df["mutAA"])
        & (~ddg_df["corrected_pos"].isin(set(protected_positions)))
    ].copy()

    dropped = before - len(ddg_df)

    if start_seq is not None:
        start_seq = clean_sequence(start_seq)
        ddg_df["matches_start_seq"] = [
            start_seq[pos - 1] == wt
            for pos, wt in zip(ddg_df["corrected_pos"], ddg_df["wtAA"])
        ]
    else:
        ddg_df["matches_start_seq"] = True

    # 如果同一个 corrected_pos / wtAA / mutAA 有重复，保留更稳定的最小 ddG
    ddg_df = ddg_df.sort_values("ddG", ascending=True)
    ddg_df = ddg_df.drop_duplicates(
        subset=["corrected_pos", "wtAA", "mutAA"],
        keep="first"
    ).reset_index(drop=True)

    ddg_lookup = {
        (int(row.corrected_pos), row.wtAA, row.mutAA): float(row.ddG)
        for row in ddg_df.itertuples(index=False)
    }

    n_beneficial = int((ddg_df["ddG"] < 0).sum())
    n_matches_start = int(ddg_df["matches_start_seq"].sum())

    print("\nThermoMPNN ddG table loaded.")
    print(f"Rows after filtering: {len(ddg_df)}")
    print(f"Dropped rows: {dropped}")
    print(f"Position correction: corrected_pos = pos + {position_shift}")
    print(f"Beneficial single mutations, ddG < 0: {n_beneficial}")
    print(f"Rows whose wtAA matches sfGFP: {n_matches_start}")

    if n_matches_start == 0:
        print(
            "[WARN] No ThermoMPNN row has wtAA matching sfGFP at corrected_pos. "
            "Please check whether the ddG file was computed on the same sequence."
        )

    return ddg_df, ddg_lookup


def sample_thermo_guided_mutation(
    seq_list,
    mutable_positions,
    protected_positions,
    ddg_df,
    max_ddg=0.0,
    temperature=0.5
):
    if ddg_df is None or len(ddg_df) == 0:
        return None

    protected_positions = set(protected_positions)
    mutable_positions = set(mutable_positions)

    rows = []
    weights = []

    for row in ddg_df.itertuples(index=False):
        pos = int(row.corrected_pos)

        if pos not in mutable_positions:
            continue

        if pos in protected_positions:
            continue

        idx = pos - 1
        current_aa = seq_list[idx]

        # 只有当前序列该位点氨基酸等于 wtAA 时，这个单点 ddG 才直接适用
        if current_aa != row.wtAA:
            continue

        if current_aa == row.mutAA:
            continue

        if max_ddg is not None and float(row.ddG) > max_ddg:
            continue

        rows.append(row)

        # ddG 越负，权重越高
        weight = np.exp(
            np.clip(
                -float(row.ddG) / max(temperature, 1e-6),
                -50,
                50
            )
        )
        weights.append(float(weight))

    if len(rows) == 0:
        return None

    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    chosen_idx = int(np.random.choice(len(rows), p=weights))
    chosen = rows[chosen_idx]

    return int(chosen.corrected_pos), chosen.mutAA, float(chosen.ddG)


def add_thermo_metrics_to_predictions(
    pred_df,
    start_seq,
    ddg_lookup,
    unscored_penalty=0.15
):
    thermo_ddg_sum = []
    thermo_ddg_mean = []
    thermo_stability_score = []
    thermo_n_scored = []
    thermo_n_unscored = []
    thermo_stabilizing_muts = []
    thermo_destabilizing_muts = []
    thermo_unscored_muts = []

    for seq in pred_df["sequence"]:
        scored_ddgs = []
        stabilizing = []
        destabilizing = []
        unscored = []

        for pos, (start_aa, query_aa) in enumerate(
            zip(start_seq, seq),
            start=1
        ):
            if start_aa == query_aa:
                continue

            mut_label = mutation_label(start_aa, pos, query_aa)
            ddg = ddg_lookup.get((pos, start_aa, query_aa))

            if ddg is None:
                unscored.append(mut_label)
            else:
                ddg = float(ddg)
                scored_ddgs.append(ddg)

                if ddg <= 0:
                    stabilizing.append(f"{mut_label}({ddg:.3f})")
                else:
                    destabilizing.append(f"{mut_label}({ddg:.3f})")

        if len(scored_ddgs) > 0:
            ddg_sum = float(np.sum(scored_ddgs))
            ddg_mean = float(np.mean(scored_ddgs))
        else:
            ddg_sum = 0.0
            ddg_mean = np.nan

        n_unscored = len(unscored)

        # ddG 越负越好，所以用 -ddG_sum
        # 未覆盖突变扣分
        stability_score = -ddg_sum - unscored_penalty * n_unscored

        thermo_ddg_sum.append(ddg_sum)
        thermo_ddg_mean.append(ddg_mean)
        thermo_stability_score.append(stability_score)
        thermo_n_scored.append(len(scored_ddgs))
        thermo_n_unscored.append(n_unscored)

        thermo_stabilizing_muts.append(
            ":".join(stabilizing) if stabilizing else ""
        )
        thermo_destabilizing_muts.append(
            ":".join(destabilizing) if destabilizing else ""
        )
        thermo_unscored_muts.append(
            ":".join(unscored) if unscored else ""
        )

    pred_df = pred_df.copy()

    pred_df["thermo_ddG_sum"] = thermo_ddg_sum
    pred_df["thermo_ddG_mean"] = thermo_ddg_mean
    pred_df["thermo_stability_score"] = thermo_stability_score
    pred_df["thermo_n_scored_mutations"] = thermo_n_scored
    pred_df["thermo_n_unscored_mutations"] = thermo_n_unscored
    pred_df["thermo_stabilizing_mutations_vs_sfGFP"] = thermo_stabilizing_muts
    pred_df["thermo_destabilizing_mutations_vs_sfGFP"] = thermo_destabilizing_muts
    pred_df["thermo_unscored_mutations_vs_sfGFP"] = thermo_unscored_muts

    return pred_df


# ============================================================
# 4. 比赛代理分数
# ============================================================

def add_competition_proxy_score(
    pred_df,
    brightness_baseline,
    low_brightness_relative_threshold=0.30,
    heat_retention_ddg_scale=1.0,
    heat_retention_floor=0.02,
    competition_proxy_weight=0.85,
    rank_tiebreaker_weight=0.15,
    aux_brightness_rank_weight=0.70,
    aux_thermo_rank_weight=0.30
):
    """
    真实比赛分数近似为：

        score = F_initial / F_initialWT * F_final / F_initial

    这里：
    - F_initial 用 predicted_brightness 近似
    - F_initialWT 用模型预测的 sfGFP 起始序列亮度近似
    - F_final / F_initial 用 ThermoMPNN ddG_sum 估计

    注意：
    这个只是搜索用代理分数，不等于真实实验分数。
    """

    pred_df = pred_df.copy()

    brightness_baseline = float(brightness_baseline)

    if not np.isfinite(brightness_baseline) or brightness_baseline <= 0:
        raise ValueError(
            f"Invalid brightness_baseline: {brightness_baseline}. "
            "The model-predicted sfGFP brightness must be positive."
        )

    pred_df["relative_initial_brightness_vs_sfGFP"] = (
        pred_df["predicted_brightness"] / brightness_baseline
    )

    # 只惩罚正 ddG_sum，即总体去稳定化突变
    ddg_positive = np.maximum(pred_df["thermo_ddG_sum"].astype(float), 0.0)

    estimated_heat_retention = np.exp(
        -ddg_positive / max(float(heat_retention_ddg_scale), 1e-6)
    )

    estimated_heat_retention = np.clip(
        estimated_heat_retention,
        float(heat_retention_floor),
        1.0
    )

    pred_df["estimated_heat_retention"] = estimated_heat_retention

    pred_df["passes_low_brightness_gate"] = (
        pred_df["relative_initial_brightness_vs_sfGFP"]
        >= float(low_brightness_relative_threshold)
    )

    pred_df["competition_proxy_score_raw"] = (
        pred_df["relative_initial_brightness_vs_sfGFP"]
        * pred_df["estimated_heat_retention"]
    )

    pred_df["competition_proxy_score"] = np.where(
        pred_df["passes_low_brightness_gate"],
        pred_df["competition_proxy_score_raw"],
        0.0
    )

    # rank 辅助项：避免早期搜索过快收敛
    pred_df["brightness_rank_pct"] = pred_df["predicted_brightness"].rank(
        method="average",
        pct=True
    )

    pred_df["thermo_rank_pct"] = pred_df["thermo_stability_score"].rank(
        method="average",
        pct=True
    )

    pred_df["auxiliary_rank_score"] = (
        float(aux_brightness_rank_weight) * pred_df["brightness_rank_pct"]
        + float(aux_thermo_rank_weight) * pred_df["thermo_rank_pct"]
    )

    total_weight = float(competition_proxy_weight) + float(rank_tiebreaker_weight)

    if total_weight <= 0:
        competition_proxy_weight = 1.0
        rank_tiebreaker_weight = 0.0
        total_weight = 1.0

    competition_proxy_weight = float(competition_proxy_weight) / total_weight
    rank_tiebreaker_weight = float(rank_tiebreaker_weight) / total_weight

    pred_df["selection_score"] = (
        competition_proxy_weight * pred_df["competition_proxy_score"]
        + rank_tiebreaker_weight * pred_df["auxiliary_rank_score"]
    )

    return pred_df


# ============================================================
# 5. 模型特征编码
# ============================================================

def encode_sequences_for_model(sequences, model_reference_seq, bundle):
    feature_names = bundle["feature_names"]
    aa_list = bundle["aa_list"]

    aa_to_idx = {aa: i for i, aa in enumerate(aa_list)}

    X = pd.DataFrame(
        np.zeros((len(sequences), len(feature_names)), dtype=np.float32),
        columns=feature_names
    )

    mutation_strings_vs_avGFP = []
    mutation_counts_vs_avGFP = []

    for row_idx, seq in enumerate(sequences):
        mut_str = sequence_to_mutation_string(model_reference_seq, seq)
        mutation_strings_vs_avGFP.append(mut_str)

        mut_count = 0

        for pos, (ref_aa, query_aa) in enumerate(
            zip(model_reference_seq, seq),
            start=1
        ):
            if ref_aa == query_aa:
                continue

            mut_count += 1

            if query_aa not in aa_to_idx:
                raise ValueError(
                    f"Invalid amino acid '{query_aa}' at position {pos}"
                )

            feature = f"pos{pos}_{query_aa}"

            if feature not in X.columns:
                raise ValueError(
                    f"Feature '{feature}' not found in model feature names. "
                    f"Please check position numbering and feature construction."
                )

            X.loc[row_idx, feature] = 1.0

        mutation_counts_vs_avGFP.append(mut_count)

        if "mutation_count" in X.columns:
            X.loc[row_idx, "mutation_count"] = mut_count

    X = X[feature_names]

    return X, mutation_strings_vs_avGFP, mutation_counts_vs_avGFP


def predict_brightness(
    sequences,
    model_reference_seq,
    start_seq,
    bundle
):
    model = bundle["model"]

    X, muts_vs_avGFP, counts_vs_avGFP = encode_sequences_for_model(
        sequences=sequences,
        model_reference_seq=model_reference_seq,
        bundle=bundle
    )

    preds = model.predict(X)

    muts_vs_sfGFP = []
    counts_vs_sfGFP = []

    for seq in sequences:
        muts_vs_sfGFP.append(sequence_to_mutation_string(start_seq, seq))
        counts_vs_sfGFP.append(count_mutations(start_seq, seq))

    result = pd.DataFrame({
        "sequence": sequences,
        "aaMutations_vs_avGFP": muts_vs_avGFP,
        "mutation_count_vs_avGFP": counts_vs_avGFP,
        "aaMutations_vs_sfGFP": muts_vs_sfGFP,
        "mutation_count_vs_sfGFP": counts_vs_sfGFP,
        "predicted_brightness": preds
    })

    return result


# ============================================================
# 6. 进化算法操作
# ============================================================

def apply_random_new_mutation(seq_list, start_seq, candidate_positions):
    available_positions = [
        pos for pos in candidate_positions
        if seq_list[pos - 1] == start_seq[pos - 1]
    ]

    if len(available_positions) == 0:
        return False

    pos = random.choice(available_positions)
    idx = pos - 1
    current_aa = seq_list[idx]

    possible_aas = [
        aa for aa in AA_LIST
        if aa != current_aa
    ]

    seq_list[idx] = random.choice(possible_aas)

    return True


def apply_thermo_or_random_new_mutation(
    seq_list,
    start_seq,
    mutable_positions,
    protected_positions,
    ddg_df=None,
    thermo_guided_prob=0.75,
    thermo_guided_max_ddg=0.0,
    thermo_sampling_temperature=0.5
):
    candidate_positions = sorted(
        list(set(mutable_positions) - set(protected_positions))
    )

    candidate_positions = [
        pos for pos in candidate_positions
        if seq_list[pos - 1] == start_seq[pos - 1]
    ]

    if len(candidate_positions) == 0:
        return False

    use_thermo = (
        ddg_df is not None
        and len(ddg_df) > 0
        and random.random() < thermo_guided_prob
    )

    if use_thermo:
        sampled = sample_thermo_guided_mutation(
            seq_list=seq_list,
            mutable_positions=candidate_positions,
            protected_positions=protected_positions,
            ddg_df=ddg_df,
            max_ddg=thermo_guided_max_ddg,
            temperature=thermo_sampling_temperature
        )

        if sampled is not None:
            pos, mut_aa, _ddg = sampled
            seq_list[pos - 1] = mut_aa
            return True

    return apply_random_new_mutation(
        seq_list=seq_list,
        start_seq=start_seq,
        candidate_positions=candidate_positions
    )


def make_single_mutant(start_seq, pos, mut_aa):
    seq_list = list(start_seq)
    seq_list[pos - 1] = mut_aa
    return "".join(seq_list)


def initialize_population(
    start_seq,
    mutable_positions,
    protected_positions,
    population_size,
    max_new_mutations_vs_start,
    ddg_df=None,
    ddg_lookup=None,
    thermo_guided_prob=0.75,
    thermo_guided_max_ddg=0.0,
    thermo_sampling_temperature=0.5,
    thermo_preseed_single_mutants=100
):
    population = {start_seq}

    protected_positions = set(protected_positions)
    mutable_positions = set(mutable_positions)

    # 先加入 ThermoMPNN 最稳定的单点突变体
    if ddg_df is not None and len(ddg_df) > 0 and thermo_preseed_single_mutants > 0:
        preseed_df = ddg_df[
            (ddg_df["corrected_pos"].isin(mutable_positions))
            & (~ddg_df["corrected_pos"].isin(protected_positions))
            & (ddg_df["matches_start_seq"])
        ].copy()

        if thermo_guided_max_ddg is not None:
            preseed_df = preseed_df[
                preseed_df["ddG"] <= thermo_guided_max_ddg
            ]

        preseed_df = preseed_df.sort_values(
            "ddG",
            ascending=True
        ).head(thermo_preseed_single_mutants)

        for row in preseed_df.itertuples(index=False):
            population.add(
                make_single_mutant(
                    start_seq,
                    int(row.corrected_pos),
                    row.mutAA
                )
            )

            if len(population) >= population_size:
                return list(population)

    while len(population) < population_size:
        seq_list = list(start_seq)

        n_mut = random.randint(1, max_new_mutations_vs_start)
        n_mut = min(n_mut, len(mutable_positions))

        for _ in range(n_mut):
            success = apply_thermo_or_random_new_mutation(
                seq_list=seq_list,
                start_seq=start_seq,
                mutable_positions=mutable_positions,
                protected_positions=protected_positions,
                ddg_df=ddg_df,
                thermo_guided_prob=thermo_guided_prob,
                thermo_guided_max_ddg=thermo_guided_max_ddg,
                thermo_sampling_temperature=thermo_sampling_temperature
            )

            if not success:
                break

        candidate = "".join(seq_list)

        candidate = force_protected_positions_to_start_seq(
            candidate_seq=candidate,
            start_seq=start_seq,
            protected_positions=protected_positions
        )

        candidate = enforce_max_new_mutations_vs_start(
            candidate_seq=candidate,
            start_seq=start_seq,
            max_new_mutations_vs_start=max_new_mutations_vs_start,
            ddg_lookup=ddg_lookup
        )

        population.add(candidate)

    return list(population)


def mutate_sequence(
    seq,
    start_seq,
    mutable_positions,
    protected_positions,
    max_new_mutations_vs_start,
    ddg_df=None,
    ddg_lookup=None,
    thermo_guided_prob=0.75,
    thermo_guided_max_ddg=0.0,
    thermo_sampling_temperature=0.5,
    min_mutations_per_step=1,
    max_mutations_per_step=3
):
    seq_list = list(seq)

    current_mut_count = count_mutations(start_seq, seq)
    remaining_capacity = max_new_mutations_vs_start - current_mut_count

    if remaining_capacity <= 0:
        return seq

    n_new = random.randint(
        min_mutations_per_step,
        max_mutations_per_step
    )

    n_new = min(n_new, remaining_capacity)

    for _ in range(n_new):
        success = apply_thermo_or_random_new_mutation(
            seq_list=seq_list,
            start_seq=start_seq,
            mutable_positions=mutable_positions,
            protected_positions=protected_positions,
            ddg_df=ddg_df,
            thermo_guided_prob=thermo_guided_prob,
            thermo_guided_max_ddg=thermo_guided_max_ddg,
            thermo_sampling_temperature=thermo_sampling_temperature
        )

        if not success:
            break

    child = "".join(seq_list)

    child = force_protected_positions_to_start_seq(
        candidate_seq=child,
        start_seq=start_seq,
        protected_positions=protected_positions
    )

    child = enforce_max_new_mutations_vs_start(
        candidate_seq=child,
        start_seq=start_seq,
        max_new_mutations_vs_start=max_new_mutations_vs_start,
        ddg_lookup=ddg_lookup
    )

    return child


def crossover_sequences(
    parent1,
    parent2,
    start_seq,
    protected_positions,
    max_new_mutations_vs_start,
    ddg_lookup=None
):
    child_list = []

    for pos, (aa1, aa2, start_aa) in enumerate(
        zip(parent1, parent2, start_seq),
        start=1
    ):
        if pos in protected_positions:
            child_list.append(start_aa)
        else:
            child_list.append(random.choice([aa1, aa2]))

    child = "".join(child_list)

    child = force_protected_positions_to_start_seq(
        candidate_seq=child,
        start_seq=start_seq,
        protected_positions=protected_positions
    )

    child = enforce_max_new_mutations_vs_start(
        candidate_seq=child,
        start_seq=start_seq,
        max_new_mutations_vs_start=max_new_mutations_vs_start,
        ddg_lookup=ddg_lookup
    )

    return child


# ============================================================
# 7. 动态进化阶段参数
# ============================================================

def get_generation_strategy(
    generation,
    n_generations,
    base_thermo_guided_prob,
    base_thermo_guided_max_ddg,
    base_thermo_sampling_temperature,
    base_elite_frac,
    base_parent_pool_frac,
    base_min_mutations_per_step,
    base_max_mutations_per_step,
    early_exploration_frac=0.30,
    late_exploit_frac=0.30
):
    progress = generation / max(n_generations, 1)

    # 早期：探索更多，热稳定约束稍微放松
    if progress <= early_exploration_frac:
        strategy = {
            "stage": "early_exploration",
            "thermo_guided_prob": max(0.20, base_thermo_guided_prob * 0.70),
            "thermo_guided_max_ddg": 0.50,
            "thermo_sampling_temperature": max(0.75, base_thermo_sampling_temperature * 1.50),
            "elite_frac": max(0.08, base_elite_frac * 0.80),
            "parent_pool_frac": min(0.60, base_parent_pool_frac * 1.30),
            "min_mutations_per_step": base_min_mutations_per_step,
            "max_mutations_per_step": base_max_mutations_per_step,
        }

    # 后期：收敛，更多选择稳定突变
    elif progress >= 1.0 - late_exploit_frac:
        strategy = {
            "stage": "late_exploitation",
            "thermo_guided_prob": min(0.85, max(base_thermo_guided_prob, 0.70)),
            "thermo_guided_max_ddg": -0.10,
            "thermo_sampling_temperature": min(0.40, base_thermo_sampling_temperature),
            "elite_frac": min(0.25, base_elite_frac * 1.30),
            "parent_pool_frac": max(0.25, base_parent_pool_frac * 0.80),
            "min_mutations_per_step": 1,
            "max_mutations_per_step": min(2, base_max_mutations_per_step),
        }

    # 中期：平衡
    else:
        strategy = {
            "stage": "middle_balanced",
            "thermo_guided_prob": base_thermo_guided_prob,
            "thermo_guided_max_ddg": base_thermo_guided_max_ddg,
            "thermo_sampling_temperature": base_thermo_sampling_temperature,
            "elite_frac": base_elite_frac,
            "parent_pool_frac": base_parent_pool_frac,
            "min_mutations_per_step": base_min_mutations_per_step,
            "max_mutations_per_step": base_max_mutations_per_step,
        }

    return strategy


# ============================================================
# 8. 主搜索函数
# ============================================================

def run_evolutionary_search(
    model_reference_seq,
    start_seq,
    model_file,
    protected_positions,
    thermompnn_ddg_file=None,
    thermompnn_position_shift=2,
    thermo_guided_mutation_prob=0.75,
    thermo_guided_max_ddg=0.0,
    thermo_sampling_temperature=0.5,
    thermo_unscored_mutation_penalty=0.15,
    thermo_preseed_single_mutants=100,
    population_size=300,
    n_generations=80,
    max_new_mutations_vs_start=8,
    elite_frac=0.15,
    parent_pool_frac=0.40,
    top_n=10,
    random_seed=42,
    low_brightness_relative_threshold=0.30,
    heat_retention_ddg_scale=1.0,
    heat_retention_floor=0.02,
    competition_proxy_weight=0.85,
    rank_tiebreaker_weight=0.15,
    aux_brightness_rank_weight=0.70,
    aux_thermo_rank_weight=0.30,
    early_exploration_frac=0.30,
    late_exploit_frac=0.30
):
    random.seed(random_seed)
    np.random.seed(random_seed)

    bundle = joblib.load(model_file)
    protein_length = bundle["protein_length"]

    model_reference_seq = clean_sequence(model_reference_seq)
    start_seq = clean_sequence(start_seq)

    validate_sequence(
        model_reference_seq,
        expected_length=protein_length,
        name="avGFP reference sequence"
    )

    validate_sequence(
        start_seq,
        expected_length=protein_length,
        name="sfGFP start sequence"
    )

    protected_positions = set(protected_positions)

    all_positions = set(range(1, protein_length + 1))
    mutable_positions = sorted(list(all_positions - protected_positions))

    if len(mutable_positions) == 0:
        raise ValueError("No mutable positions available.")

    ddg_df, ddg_lookup = load_thermompn_ddg_table(
        ddg_file=thermompnn_ddg_file,
        protein_length=protein_length,
        protected_positions=protected_positions,
        position_shift=thermompnn_position_shift,
        start_seq=start_seq
    )

    if len(ddg_lookup) == 0:
        thermo_guided_mutation_prob = 0.0

    # 预测 sfGFP 起始序列亮度，用作 F_initialWT 近似值
    sf_baseline_df = predict_brightness(
        sequences=[start_seq],
        model_reference_seq=model_reference_seq,
        start_seq=start_seq,
        bundle=bundle
    )

    sf_baseline_brightness = float(sf_baseline_df.loc[0, "predicted_brightness"])

    print("Model loaded successfully.")
    print(f"Protein length: {protein_length}")
    print(f"Protected positions: {len(protected_positions)}")
    print(f"Mutable positions: {len(mutable_positions)}")
    print(f"sfGFP model-predicted baseline brightness: {sf_baseline_brightness:.6f}")
    print(f"Low brightness gate: {low_brightness_relative_threshold:.2f} x sfGFP")
    print(f"Thermo-guided mutation probability: {thermo_guided_mutation_prob}")

    sf_vs_av_mutations = sequence_to_mutation_string(
        model_reference_seq,
        start_seq
    )

    sf_vs_av_count = count_mutations(
        model_reference_seq,
        start_seq
    )

    print("\nsfGFP relative to avGFP:")
    print(f"Mutation count: {sf_vs_av_count}")
    print(f"Mutations: {sf_vs_av_mutations}")

    population = initialize_population(
        start_seq=start_seq,
        mutable_positions=mutable_positions,
        protected_positions=protected_positions,
        population_size=population_size,
        max_new_mutations_vs_start=max_new_mutations_vs_start,
        ddg_df=ddg_df,
        ddg_lookup=ddg_lookup,
        thermo_guided_prob=thermo_guided_mutation_prob,
        thermo_guided_max_ddg=thermo_guided_max_ddg,
        thermo_sampling_temperature=thermo_sampling_temperature,
        thermo_preseed_single_mutants=thermo_preseed_single_mutants
    )

    global_history = []

    for generation in range(1, n_generations + 1):

        strategy = get_generation_strategy(
            generation=generation,
            n_generations=n_generations,
            base_thermo_guided_prob=thermo_guided_mutation_prob,
            base_thermo_guided_max_ddg=thermo_guided_max_ddg,
            base_thermo_sampling_temperature=thermo_sampling_temperature,
            base_elite_frac=elite_frac,
            base_parent_pool_frac=parent_pool_frac,
            base_min_mutations_per_step=MIN_MUTATIONS_PER_STEP,
            base_max_mutations_per_step=MAX_MUTATIONS_PER_STEP,
            early_exploration_frac=early_exploration_frac,
            late_exploit_frac=late_exploit_frac
        )

        pred_df = predict_brightness(
            sequences=population,
            model_reference_seq=model_reference_seq,
            start_seq=start_seq,
            bundle=bundle
        )

        pred_df = add_thermo_metrics_to_predictions(
            pred_df=pred_df,
            start_seq=start_seq,
            ddg_lookup=ddg_lookup,
            unscored_penalty=thermo_unscored_mutation_penalty
        )

        pred_df = add_competition_proxy_score(
            pred_df=pred_df,
            brightness_baseline=sf_baseline_brightness,
            low_brightness_relative_threshold=low_brightness_relative_threshold,
            heat_retention_ddg_scale=heat_retention_ddg_scale,
            heat_retention_floor=heat_retention_floor,
            competition_proxy_weight=competition_proxy_weight,
            rank_tiebreaker_weight=rank_tiebreaker_weight,
            aux_brightness_rank_weight=aux_brightness_rank_weight,
            aux_thermo_rank_weight=aux_thermo_rank_weight
        )

        pred_df["generation"] = generation
        pred_df["search_stage"] = strategy["stage"]

        pred_df = pred_df.sort_values(
            [
                "selection_score",
                "competition_proxy_score",
                "predicted_brightness",
                "thermo_stability_score"
            ],
            ascending=[False, False, False, False]
        ).reset_index(drop=True)

        best_selection_score = pred_df.loc[0, "selection_score"]
        best_competition_proxy_score = pred_df.loc[0, "competition_proxy_score"]
        best_relative_brightness = pred_df.loc[0, "relative_initial_brightness_vs_sfGFP"]
        best_heat_retention = pred_df.loc[0, "estimated_heat_retention"]
        best_brightness = pred_df.loc[0, "predicted_brightness"]
        best_thermo_score = pred_df.loc[0, "thermo_stability_score"]
        best_thermo_ddg_sum = pred_df.loc[0, "thermo_ddG_sum"]
        best_muts_vs_sf = pred_df.loc[0, "aaMutations_vs_sfGFP"]
        best_muts_vs_av = pred_df.loc[0, "aaMutations_vs_avGFP"]

        print(
            f"Generation {generation:03d} | "
            f"stage = {strategy['stage']} | "
            f"selection_score = {best_selection_score:.4f} | "
            f"competition_proxy = {best_competition_proxy_score:.4f} | "
            f"rel_brightness = {best_relative_brightness:.3f} | "
            f"heat_retention = {best_heat_retention:.3f} | "
            f"brightness = {best_brightness:.6f} | "
            f"thermo_score = {best_thermo_score:.3f} | "
            f"thermo_ddG_sum = {best_thermo_ddg_sum:.3f} | "
            f"vs_sfGFP: {best_muts_vs_sf} | "
            f"vs_avGFP: {best_muts_vs_av}"
        )

        global_history.append(pred_df)

        n_elite = max(1, int(population_size * strategy["elite_frac"]))
        n_parent_pool = max(2, int(population_size * strategy["parent_pool_frac"]))

        elites = pred_df.head(n_elite)["sequence"].tolist()
        parent_pool = pred_df.head(n_parent_pool)["sequence"].tolist()

        next_population = set(elites)

        while len(next_population) < population_size:
            parent1, parent2 = random.sample(parent_pool, 2)

            child = crossover_sequences(
                parent1=parent1,
                parent2=parent2,
                start_seq=start_seq,
                protected_positions=protected_positions,
                max_new_mutations_vs_start=max_new_mutations_vs_start,
                ddg_lookup=ddg_lookup
            )

            child = mutate_sequence(
                seq=child,
                start_seq=start_seq,
                mutable_positions=mutable_positions,
                protected_positions=protected_positions,
                max_new_mutations_vs_start=max_new_mutations_vs_start,
                ddg_df=ddg_df,
                ddg_lookup=ddg_lookup,
                thermo_guided_prob=strategy["thermo_guided_prob"],
                thermo_guided_max_ddg=strategy["thermo_guided_max_ddg"],
                thermo_sampling_temperature=strategy["thermo_sampling_temperature"],
                min_mutations_per_step=strategy["min_mutations_per_step"],
                max_mutations_per_step=strategy["max_mutations_per_step"]
            )

            next_population.add(child)

        population = list(next_population)

    all_results = pd.concat(global_history, axis=0)

    all_results = all_results.drop_duplicates(
        subset=["sequence"]
    ).sort_values(
        [
            "selection_score",
            "competition_proxy_score",
            "predicted_brightness",
            "thermo_stability_score"
        ],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    top_results = all_results.head(top_n).copy()

    return top_results, all_results


# ============================================================
# 9. 运行进化搜索
# ============================================================

if __name__ == "__main__":

    top10_df, all_results_df = run_evolutionary_search(
        model_reference_seq=AVGFP_WT_SEQUENCE,
        start_seq=SFGFP_START_SEQUENCE,
        model_file=MODEL_FILE,
        protected_positions=PROTECTED_POSITIONS,
        thermompnn_ddg_file=THERMOMPNN_DDG_FILE,
        thermompnn_position_shift=THERMOMPNN_POSITION_SHIFT,
        thermo_guided_mutation_prob=THERMO_GUIDED_MUTATION_PROB,
        thermo_guided_max_ddg=THERMO_GUIDED_MAX_DDG,
        thermo_sampling_temperature=THERMO_SAMPLING_TEMPERATURE,
        thermo_unscored_mutation_penalty=THERMO_UNSCORED_MUTATION_PENALTY,
        thermo_preseed_single_mutants=THERMO_PRESEED_SINGLE_MUTANTS,
        population_size=POPULATION_SIZE,
        n_generations=N_GENERATIONS,
        max_new_mutations_vs_start=MAX_NEW_MUTATIONS_VS_SFGFP,
        elite_frac=ELITE_FRAC,
        parent_pool_frac=PARENT_POOL_FRAC,
        top_n=TOP_N,
        random_seed=RANDOM_SEED,
        low_brightness_relative_threshold=LOW_BRIGHTNESS_RELATIVE_THRESHOLD,
        heat_retention_ddg_scale=HEAT_RETENTION_DDG_SCALE,
        heat_retention_floor=HEAT_RETENTION_FLOOR,
        competition_proxy_weight=COMPETITION_PROXY_WEIGHT,
        rank_tiebreaker_weight=RANK_TIEBREAKER_WEIGHT,
        aux_brightness_rank_weight=AUX_BRIGHTNESS_RANK_WEIGHT,
        aux_thermo_rank_weight=AUX_THERMO_RANK_WEIGHT,
        early_exploration_frac=EARLY_EXPLORATION_FRAC,
        late_exploit_frac=LATE_EXPLOIT_FRAC
    )

    top10_df.to_csv(
        "top20_max_mutaion_6_competition_proxy_thermo_guided_evolved_sequences_from_sfGFP.csv",
        index=False
    )

    all_results_df.to_csv(
        "max_mutaion_6_all_competition_proxy_thermo_guided_evolutionary_search_results_from_sfGFP.csv",
        index=False
    )

    output_columns = [
        "selection_score",
        "competition_proxy_score",
        "competition_proxy_score_raw",
        "relative_initial_brightness_vs_sfGFP",
        "estimated_heat_retention",
        "passes_low_brightness_gate",
        "predicted_brightness",
        "thermo_stability_score",
        "thermo_ddG_sum",
        "thermo_ddG_mean",
        "thermo_n_scored_mutations",
        "thermo_n_unscored_mutations",
        "mutation_count_vs_sfGFP",
        "mutation_count_vs_avGFP",
        "aaMutations_vs_sfGFP",
        "thermo_stabilizing_mutations_vs_sfGFP",
        "thermo_destabilizing_mutations_vs_sfGFP",
        "thermo_unscored_mutations_vs_sfGFP",
        "aaMutations_vs_avGFP",
        "generation",
        "search_stage",
        "sequence"
    ]

    print("\nTop evolved sequences, sorted by selection_score:")
    print(top10_df[output_columns])

    print("\nSaved files:")
    print("top10_competition_proxy_thermo_guided_evolved_sequences_from_sfGFP.csv")
    print("all_competition_proxy_thermo_guided_evolutionary_search_results_from_sfGFP.csv")