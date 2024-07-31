import lib.algorithms
import lib.ml
import numpy as np
from datetime import datetime
import os

def main(
    n_traces,
    n_timesteps,
    merge_state_labels,
    labels_to_binary,
    balance_classes,
    outdir,
    reduce_memory,
):
    """
    Parameters
    ----------
    n_traces:
        Number of traces to generate
    n_timesteps:
        Length of each trace
    merge_state_labels:
        Whether to merge all HMM states above 2 into "dynamic", as the HMM
        predictions don't work so well yet
    labels_to_binary:
        Whether to convert all labels to smFRET/not-smFRET (for each frame)
    balance_classes:
        Whether to balance classes based on the distribution of frame 1 (as
        this changes over time due to bleaching)
    outdir:
        Output directory
    """
   
    X, matrices = lib.algorithms.generate_traces(
        n_traces=int(n_traces),
        state_means=(0.45, 0.7),
        random_k_states_max=2,
        min_state_diff=0.15,
        D_lifetime=150,
        A_lifetime=150,
        blink_prob=0.3,
        bleed_through=(0, 0.1),
        aa_mismatch=(-0.35, 0.35),
        trace_length=n_timesteps,
        trans_prob=(0.0, 0.40),
        noise=(0.1, 0.30),
        trans_mat=None,
        au_scaling_factor=1,
        aggregation_prob=0.05,
        max_aggregate_size=3,
        null_fret_value=-1,
        acceptable_noise=0.35,
        S_range=(0.3, 0.7),
        scramble_prob=0.25,
        gamma_noise_prob=0.8,
        falloff_lifetime=500,
        falloff_prob=0.05,
        merge_labels=False,
        discard_unbleached=False,
        progressbar_callback=None,
        callback_every=1,
        return_matrix=True,
        run_headless_parallel=True,
        scramble_decouple_prob=0.9,
        reduce_memory=reduce_memory,
        merge_state_labels=merge_state_labels,
    )
    
    labels = X["label"].values

    if reduce_memory:
        X = X[["D-Dexc-rw", "A-Dexc-rw", "A-Aexc-rw"]].values
    else:
        X = X[["D-Dexc-rw", "A-Dexc-rw", "A-Aexc-rw", "E", "E_true"]].values

    if np.any(X == -1):
        print(
            "Dataset contains negative E_true. Be careful if using this "
            "for regression!"
        )

    X, labels = lib.ml.preprocess_2d_timeseries_seq2seq(
        X=X, y=labels, n_timesteps=n_timesteps
    )
    print("Before balance: ", set(labels.ravel()))
    ext = False

    if labels_to_binary:
        labels = lib.ml.labels_to_binary(
            labels, one_hot=False, to_ones=(4, 5, 6, 7, 8)
        )
        ext = "_binary"
        print("After binarize ", set(labels.ravel()))

    if balance_classes:
        X, labels = lib.ml.balance_classes(
            X, labels, exclude_label_from_limiting=0, frame=0
        )
        print("After balance:  ", set(labels.ravel()))

    os.makedirs(outdir, exist_ok=True)
    now = datetime.now().strftime("%d%H%M")
    data_id = np.array(int(now))
    print(now)
    np.save(outdir + "/data_id.npy", data_id)
    np.save(outdir + "/x_sim_" + now + ".npy", X)
    np.save(outdir + "/y_sim_" + now + ".npy", labels)


    print(X.shape)
    print(labels.shape)
    print("Generated {} traces".format(X.shape[0]))

if __name__ == "__main__":
    main(
        n_traces=1000,
        n_timesteps=800,
        merge_state_labels=True,
        balance_classes=True,
        labels_to_binary=False,
        reduce_memory=True,
        outdir="data",
    )
