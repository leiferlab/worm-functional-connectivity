#!/usr/bin/env python
#

import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np

import wormfunconn as wfc


def use_mock_atlas_scalar(
    mode="amplitude",
    amplitude_threshold=1.2,
    show_plot=False,
    matrix_output=None,
    pairwise_output=None,
    loglevel=logging.INFO,
):

    logging.basicConfig(format="%(levelname)s: %(message)s", level=loglevel)

    # Get atlas folder
    folder = os.path.join(os.path.dirname(__file__), "../atlas/")

    # Create FunctionalAtlas instance from file
    funatlas = wfc.FunctionalAtlas.from_file(folder, "mock")

    # Get scalar version of functional connectivity
    s_fconn = funatlas.get_scalar_connectivity(
        mode=mode, threshold={"amplitude": amplitude_threshold}
    )

    if show_plot:
        # Plot
        plt.imshow(s_fconn, cmap="coolwarm", vmin=-1, vmax=1)
        plt.show()

    if matrix_output:
        np.savetxt(matrix_output, s_fconn, delimiter="\t")

    # TODO Ensure that these labels are correct
    # i.e. Is the order of rows and columns the same as the order of the neu_ids?
    if pairwise_output:
        with open(pairwise_output, "w") as pairwise_file:
            for (xidx, xlab) in enumerate(funatlas.neu_ids):
                for (yidx, ylab) in enumerate(funatlas.neu_ids):
                    pairwise_file.write(f"{xlab}\t{ylab}\t{s_fconn[xidx,yidx]}\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Use the mock atlas and return a scalar version of "
            "the functional connectivity matrix (amplitude, timescales, "
            "or other)"
        ),
        epilog=(
            "As an alternative to the commandline, params can be "
            "placed in a file, one per line, and specified on the "
            "commandline like '%(prog)s @params.conf'."
        ),
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--mode",
        help=(
            "Type of scalar quantity to extract from functional connectivity. "
            "Can be amplitude or timescales."
        ),
        default="amplitude",
        choices=["amplitude", "timescale_1"],
        metavar="MODE",
    )
    parser.add_argument(
        "--amplitude-threshold",
        help=("Amplitude values below the threshold will be set to zero."),
        type=float,
        default=1.2,
        metavar="THRESHOLD",
    )
    parser.add_argument(
        "--matrix-output",
        help="Save the matrix to a tab delimited file",
        metavar="FILE",
    )
    parser.add_argument(
        "--pairwise-output",
        help="Save the matrix with one line per neuron pair in a tab delimited file",
        metavar="FILE",
    )
    parser.add_argument("-p", "--show-plot", help="Plot output", action="store_true")
    parser.add_argument(
        "-v", "--verbose", help="increase output verbosity", action="store_true"
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {wfc.__version__}"
    )
    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    use_mock_atlas_scalar(
        mode=args.mode,
        amplitude_threshold=args.amplitude_threshold,
        show_plot=args.show_plot,
        matrix_output=args.matrix_output,
        pairwise_output=args.pairwise_output,
        loglevel=loglevel,
    )


if __name__ == "__main__":
    main()
