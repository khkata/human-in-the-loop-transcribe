#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.config import TEST_INTEGRATED_CSV

def compute_cost_wer(df: pd.DataFrame, rank_col: str, k_values: list[int], human_cost: float):
    costs = []
    wers = []
    for k in k_values:
        if k == 0:
            assigned_idx = []
        elif rank_col == 'avg_logprob':
            assigned_idx = df.nsmallest(k, rank_col).index
        else:
            assigned_idx = df.nlargest(k, rank_col).index
        cost = k * human_cost
        wer_values = np.where(
            df.index.isin(assigned_idx),
            df['human_wer'],
            df['asr_wer']
        )
        overall_wer = np.mean(wer_values)
        costs.append(cost)
        wers.append(overall_wer)
    return costs, wers


def main(args):
    # データ読み込み
    df = pd.read_csv(args.integrated_csv)
    # 各種設定
    human_cost = args.human_cost
    max_k = len(df)
    k_values = list(range(0, max_k + 1))

    # 曲線データ計算
    cost_low_lp, wer_low_lp = compute_cost_wer(df, 'avg_logprob', k_values, human_cost)
    cost_high_wp, wer_high_wp = compute_cost_wer(df, 'wer_pred',    k_values, human_cost)

    # プロット
    plt.figure(figsize=(8, 5))
    plt.plot(cost_low_lp, wer_low_lp,  label='avg_logprob')
    plt.plot(cost_high_wp, wer_high_wp, label='wer_pred')

    # 基準点
    cost_asr_only   = 0
    wer_asr_only    = df['asr_wer'].mean()
    N = len(df)
    cost_human_only = N * human_cost
    wer_human_only  = df['human_wer'].mean()

    plt.scatter(cost_asr_only, wer_asr_only, marker='s', s=60, label='All ASR', zorder=5)
    plt.scatter(cost_human_only, wer_human_only, marker='s', s=60, label='All Human', zorder=5)

    plt.annotate(f"({cost_asr_only:.0f}, {wer_asr_only:.3f})",
                 xy=(cost_asr_only, wer_asr_only), xytext=(8, 0), textcoords='offset points')
    plt.annotate(f"({cost_human_only:.0f}, {wer_human_only:.3f})",
                 xy=(cost_human_only, wer_human_only), xytext=(0, 8), textcoords='offset points')

    plt.title('Cost-WER Curve')
    plt.xlabel('Cost')
    plt.ylabel('WER')
    plt.legend()
    plt.grid(True)

    # 保存
    out_dir = args.output_dir or os.path.dirname(args.integrated_csv)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.output_filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {out_path}")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot cost-WER curve for ASR/human integration')
    parser.add_argument('--integrated_csv', default=TEST_INTEGRATED_CSV,
                        help='Path to integrated CSV with asr/human columns')
    parser.add_argument('--human_cost', type=float, default=15.0,
                        help='Cost per utterance for human transcription')
    parser.add_argument('--output_dir', default=None,
                        help='Directory to save the plot (default: same dir as integrated_csv)')
    parser.add_argument('--output_filename', default='cost_wer_curve.png',
                        help='Filename for the saved plot')
    args = parser.parse_args()
    main(args)