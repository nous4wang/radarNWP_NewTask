# E45 -- the arm table, in ONE place.
#
# Why this file exists: the training script and the evaluation script
# both have to know which arm a given index is, and what that arm's
# recipe is.  When that mapping was written twice, the two copies
# drifted -- and the drift was silent, because a wrong arm still trains
# and still scores.
#
# The index is the Slurm ARRAY TASK ID.  The table is:
#
#   index  arm  model                 loss      extra loss flags
#   0      B0   simvp_v2              balanced  (none)
#   1      A1   amp_monotone_gate     balanced  (none)
#   2      A2   amp_hard_support      balanced  (none)
#   3      A3a  topo_simvp            topo      --topo-clamp-mm 20
#   4      A3b  topo_simvp            topo      --topo-truth-smooth-sigma 1.0
#                                                  --topo-area-normalize
#   5      A4   scaled_structure      balanced  (none)
#   6      A4b  scaled_structure_own  balanced  (none)
#
# WHY THE TOPOLOGY FLAGS ARE HARDCODED HERE
# -----------------------------------------
# E45's first attempt passed them through environment variables that the
# submitter never exported.  The training script therefore received 0.0
# for both, and 0.0 means "off": A3a ran with NO amplitude cap and A3b
# ran with NO truth smoothing, so neither arm tested the mechanism its
# name claims.  The flags are now part of the arm's definition, in the
# same file as the arm's name, so a missing export can no longer
# silently turn a mechanism off.
#
# The values are the pre-registered ones (E45 section 2.5.1): A3a adds
# ONLY the amplitude cap, A3b adds ONLY the truth smoothing and the
# area normalisation, and each therefore differs from E44's A3 in
# exactly one thing.
#
# The fifth column is the PROBE's inline arm option, because the
# mechanism probe rebuilds each arm's own loss term and must be told
# which flags that arm was trained with.  Its syntax is the probe's:
# "key=value" pairs separated by commas (NAME|key=value,key=value).
#
# Each row is five pipe-separated fields: arm, model, loss, train
# flags, probe flags.  The train-flags field holds space-separated
# command-line arguments and is read with "read -r -a", so it must not
# contain quotes.

E45_ARM_TABLE=(
    "B0|simvp_v2|balanced||"
    "A1|amp_monotone_gate|balanced||"
    "A2|amp_hard_support|balanced||"
    "A3a|topo_simvp|topo|--topo-clamp-mm 20|topo_clamp_mm=20.0"
    "A3b|topo_simvp|topo|--topo-truth-smooth-sigma 1.0 --topo-area-normalize|topo_truth_smooth_sigma=1.0,topo_area_normalize=1"
    "A4|scaled_structure|balanced||"
    "A4b|scaled_structure_own|balanced||"
)

#: How many arms there are.  Both scripts check the index against it and
#: refuse to run an index they do not have.
#:
#: The count is written out rather than taken as ${#E45_ARM_TABLE[@]}:
#: the bash that the local test suite starts (the PortableGit one) is
#: old enough to report the length of an array assigned in one statement
#: as zero, even though the array itself holds all seven rows.  Counting
#: the rows here keeps the number right on every bash, and the remote
#: job's own bash is not the one that would disagree.
E45_N_ARMS=7

# e45_row <index> -> the whole row
e45_row() {
    local idx="$1"
    if [[ -z "${idx}" || "${idx}" -lt 0 || "${idx}" -ge "${E45_N_ARMS}" ]]; then
        return 1
    fi
    printf '%s' "${E45_ARM_TABLE[${idx}]}"
}

# e45_field <index> <column 1..5> -> one field of the row
e45_field() {
    local row rest field
    row="$(e45_row "$1")" || return 1
    field="$2"
    rest="${row}"
    local i
    for (( i = 1; i < field; i++ )); do
        rest="${rest#*|}"
    done
    if [[ "${field}" -lt 5 ]]; then
        rest="${rest%%|*}"
    fi
    printf '%s' "${rest}"
}

# e45_arm <index> -> the arm's name
e45_arm() { e45_field "$1" 1; }

# e45_model <index> -> the model name
e45_model() { e45_field "$1" 2; }

# e45_loss <index> -> the loss name
e45_loss() { e45_field "$1" 3; }
