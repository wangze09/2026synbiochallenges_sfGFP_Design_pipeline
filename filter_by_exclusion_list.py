"""
filter_by_exclusion_list.py

将进化搜索 Top 候选序列与 FPbase 数据库排除列表比对，
过滤掉已存在于数据库中的序列，输出真正新颖的候选序列。

输入文件：
  - Exclusion_List.csv           FPbase 数据库全量序列（列名：Sequence）
  - top20_max_mutaion_6_...csv   max_mutations=6 的 Top20 进化结果
  - top20_max_mutation_8_...csv  max_mutations=8 的 Top20 进化结果

输出文件：
  - top20_max_mutaion_6_..._novel.csv
  - top20_max_mutation_8_..._novel.csv
"""

import pandas as pd

# ============================================================
# 参数配置
# ============================================================

EXCLUSION_LIST_FILE = "Exclusion_List.csv"
EXCLUSION_SEQ_COL = "Sequence"          # Exclusion_List 中序列的列名

INPUT_FILES = [
    "top20_max_mutaion_6_competition_proxy_thermo_guided_evolved_sequences_from_sfGFP.csv",
    "top20_max_mutation_8_competition_proxy_thermo_guided_evolved_sequences_from_sfGFP.csv",
]

EVOLVED_SEQ_COL = "sequence"            # 进化结果文件中序列的列名

# ============================================================
# 主流程
# ============================================================

print("Loading exclusion list ...")
ex_df = pd.read_csv(EXCLUSION_LIST_FILE)

if EXCLUSION_SEQ_COL not in ex_df.columns:
    raise ValueError(
        f"Column '{EXCLUSION_SEQ_COL}' not found in {EXCLUSION_LIST_FILE}. "
        f"Available columns: {ex_df.columns.tolist()}"
    )

exclusion_set = set(ex_df[EXCLUSION_SEQ_COL].str.strip().str.upper())
print(f"Exclusion list loaded: {len(exclusion_set)} unique sequences\n")

for input_file in INPUT_FILES:
    print(f"Processing: {input_file}")

    df = pd.read_csv(input_file)

    if EVOLVED_SEQ_COL not in df.columns:
        raise ValueError(
            f"Column '{EVOLVED_SEQ_COL}' not found in {input_file}. "
            f"Available columns: {df.columns.tolist()}"
        )

    n_before = len(df)

    is_novel = ~df[EVOLVED_SEQ_COL].str.strip().str.upper().isin(exclusion_set)
    df_novel = df[is_novel].reset_index(drop=True)

    n_after = len(df_novel)
    n_removed = n_before - n_after

    print(f"  Before filtering : {n_before} sequences")
    print(f"  After filtering  : {n_after} sequences")
    print(f"  Removed (in FPbase): {n_removed} sequences")

    # 输出文件名：在原文件名末尾加 _novel
    output_file = input_file.replace(".csv", "_novel.csv")
    df_novel.to_csv(output_file, index=False)
    print(f"  Saved -> {output_file}")

    if n_after > 0:
        print(f"\n  Top 5 novel sequences (by selection_score):")
        display_cols = [
            "aaMutations_vs_sfGFP",
            "mutation_count_vs_sfGFP",
            "predicted_brightness",
            "competition_proxy_score",
            "selection_score",
        ]
        display_cols = [c for c in display_cols if c in df_novel.columns]
        print(df_novel[display_cols].head(5).to_string(index=False))
    print()

print("Done.")
