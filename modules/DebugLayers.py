"""
Layers with the sole purpose of debugging.
These should not be included in 'real' models
"""

import os
from datetime import datetime, timedelta
import tensorflow as tf
import plotly.express as px
import pandas as pd
import numpy as np

from DeepJetCore.training.DeepJet_callbacks import publish as dcj_publish
from DeepJetCore.wandb_interface import wandb_wrapper
from plotting_tools import shuffle_truth_colors
from oc_helper_ops import SelectWithDefault


class _Publish:

    def __init__(self, time_threshold=3600):
        self.time_threshold = timedelta(seconds=time_threshold)
        self.created_time = datetime.now()
        self.last_logged_time = {}

    def publish(self, infile, where_to, ftype="html"):
        if where_to == "wandb":
            if not wandb_wrapper.active:
                return
            if ftype != "html":
                print("Warning: Unsupported file type. Only 'html' is supported.")
                return

            # Strip name of path and extension
            infilename = os.path.basename(infile)
            infilename = os.path.splitext(infilename)[0]

            # Read HTML content from file
            with open(infile, "r") as f:
                html_content = f.read()

            self.log_html_to_wandb(infilename, html_content)
        else:
            dcj_publish(infile, where_to)

    def log_html_to_wandb(self, infilename, html_content):
        current_time = datetime.now()
        last_time = self.last_logged_time.get(infilename, datetime.min)

        def delete_old():
            wandb = wandb_wrapper.wandb()
            # get the current run

            run = wandb.Api().run(wandb.run.path)
            files = run.files()
            for file in files:
                namewithouthash = file.name[
                    : -(2 + 1 + 20 + 1 + 4)
                ]  # additional _X_20hash.html
                # remove the prepending path if any
                namewithouthash = os.path.basename(namewithouthash)
                if infilename == namewithouthash:
                    file.delete()

        if current_time - last_time >= self.time_threshold:  # DEBUG
            # Log HTML content to wandb
            # add wandb step number to name
            delete_old()
            wandb_wrapper.log({infilename: wandb_wrapper.wandb().Html(html_content)})
            self.last_logged_time[infilename] = current_time
        else:
            next_allowed_time = last_time + self.time_threshold
            time_remaining = (next_allowed_time - current_time).total_seconds()
            print(
                f"Warning: {infilename} was logged less than {self.time_threshold} s ago. Next allowed logging time in {time_remaining // 60:.0f} minutes."
            )


# Usage example:
publisher = _Publish(time_threshold=1800)  # allow every half an hour


# Replace the original function call with a method call on the publisher instance
def publish(infile, where_to, ftype="html"):
    publisher.publish(infile, where_to, ftype)


class CumulativeArray(object):
    def __init__(self, capacity=60, default=0.0, name=None):

        assert capacity > 0
        self.data = None
        self.capacity = capacity
        self.default = default
        self.name = name

    def put(self, arr):
        arr = np.where(arr == np.nan, self.default, arr)
        if self.data is None:
            self.data = [np.array(arr)]
        else:
            self.data.append(np.array(arr))
            if len(self.data) > self.capacity:
                self.data = self.data[1:]  # remove oldest

    def get(self):
        return np.sum(self.data, axis=0)


class _DebugPlotBase(tf.keras.layers.Layer):
    def __init__(
        self,
        plot_every: int,
        outdir: str = "",
        plot_only_training=True,
        publish=None,
        externally_triggered=False,
        **kwargs,
    ):

        if "dynamic" in kwargs:
            super(_DebugPlotBase, self).__init__(**kwargs)
        else:
            super(_DebugPlotBase, self).__init__(dynamic=False, **kwargs)

        self.plot_every = plot_every
        self.externally_triggered = externally_triggered
        self.triggered = False
        self.plot_only_training = plot_only_training
        if len(outdir) < 1:
            self.plot_every = 0
        self.outdir = outdir
        self.counter = 0
        if not os.path.isdir(os.path.dirname(self.outdir)):  # could not be created
            self.outdir = ""

        self.publish = publish

    def get_config(self):
        config = {
            "plot_every": self.plot_every,
            "outdir": self.outdir,
            "publish": self.publish,
            "externally_triggered": self.externally_triggered,
        }
        base_config = super(_DebugPlotBase, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shapes):
        return input_shapes[0]

    def build(self, input_shape):
        super(_DebugPlotBase, self).build(input_shape)

    def plot(self, inputs, training=None):
        raise ValueError(
            "plot(self, inputs, training=None) needs to be implemented in inheriting class"
        )

    def create_base_output_path(self):
        return self.outdir + "/" + self.name

    def check_make_plot(self, inputs, training=None):

        if self.externally_triggered:
            return self.triggered

        out = inputs
        if isinstance(inputs, list):
            out = inputs[0]
        if (
            training is None or training == False
        ) and self.plot_only_training:  # only run in training mode
            return False

        if (
            len(inputs[0].shape) < 1
            or inputs[0].shape[0] is None
            or inputs[0].shape[0] == 0
        ):
            return False

        if self.plot_every <= 0:
            return False
        if not hasattr(out, "numpy"):  # only in eager
            return False

        # plot initial state
        if self.counter >= 0 and self.counter < self.plot_every:
            self.counter += 1
            return False

        if len(self.outdir) < 1:
            return False

        # only now plot
        self.counter = 0
        return True

    def call(self, inputs, training=None):
        out = inputs
        if isinstance(inputs, list):
            out = inputs[0]
        self.add_loss(0.0 * tf.reduce_sum(out[0]))  # to keep it alive

        if not self.check_make_plot(inputs, training):
            return out

        os.system("mkdir -p " + self.outdir)
        try:
            print(self.name, "plotting...")
            self.plot(inputs, training)
        except Exception as e:
            print(e)
            # do nothing, don't interrupt training because a debug plot failed

        return out


def switch_off_debug_plots(keras_model):
    for l in keras_model.layers:
        if isinstance(l, _DebugPlotBase):
            l.plot_every = -1
    return keras_model


class PlotCoordinates(_DebugPlotBase):

    def __init__(self, no_noise_plot=False, **kwargs):
        """
        Options
            - no_noise_plot: also plot without noise
        Takes as input
         - coordinate
         - features (first will be used for size)
         - truth indices
         - row splits

        Returns coordinates (unchanged)
        """
        super(PlotCoordinates, self).__init__(**kwargs)
        self.no_noise_plot = no_noise_plot

    def get_config(self):
        config = {"no_noise_plot": self.no_noise_plot}
        base_config = super(PlotCoordinates, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def plot(self, inputs, training=None):

        coords, features, hoverfeat, nidx, tidx, rs = 6 * [None]
        if len(inputs) == 4:
            coords, features, tidx, rs = inputs
        elif len(inputs) == 5:
            coords, features, hoverfeat, tidx, rs = inputs
        elif len(inputs) == 6:
            coords, features, hoverfeat, nidx, tidx, rs = inputs

        # give each an index
        idxs = np.arange(features.shape[0])

        # just select first
        coords = coords[0 : rs[1]]
        tidx = tidx[0 : rs[1]]
        idxs = idxs[0 : rs[1]]
        if len(tidx.shape) < 2:
            tidx = tidx[..., tf.newaxis]
        features = features[0 : rs[1]]
        if hoverfeat is not None:
            hoverfeat = hoverfeat[0 : rs[1]]
            hoverfeat = hoverfeat.numpy()

        if nidx is not None:
            nidx = nidx[0 : rs[1]]
            n_tidxs = SelectWithDefault(nidx, tidx, -2)
            n_sameasprobe = tf.cast(
                tf.expand_dims(tidx, axis=2) == n_tidxs[:, 1:, :], dtype="float32"
            )
            av_same = tf.reduce_mean(n_sameasprobe, axis=1)  # V x 1

        if coords.shape[1] > 2:
            # just project
            for i in range(coords.shape[1] - 2):
                data = {
                    "X": coords[:, 0 + i : 1 + i].numpy(),
                    "Y": coords[:, 1 + i : 2 + i].numpy(),
                    "Z": coords[:, 2 + i : 3 + i].numpy(),
                    "tIdx": tidx[:, 0:1].numpy(),
                    "features": features[:, 0:1].numpy(),
                    "idx": idxs[..., np.newaxis],
                }
                hoverdict = {}
                if hoverfeat is not None:
                    for j in range(hoverfeat.shape[1]):
                        hoverdict["f_" + str(j)] = hoverfeat[:, j : j + 1]
                    data.update(hoverdict)

                if nidx is not None:
                    data.update({"av_same": av_same})

                df = pd.DataFrame(
                    np.concatenate([data[k] for k in data], axis=1),
                    columns=[k for k in data],
                )
                df["orig_tIdx"] = df["tIdx"]
                rdst = np.random.RandomState(1234567890)  # all the same
                shuffle_truth_colors(df, "tIdx", rdst)

                hover_data = ["orig_tIdx", "idx"] + [k for k in hoverdict.keys()]
                if nidx is not None:
                    hover_data.append("av_same")
                fig = px.scatter_3d(
                    df,
                    x="X",
                    y="Y",
                    z="Z",
                    color="tIdx",
                    size="features",
                    hover_data=hover_data,
                    template="plotly_dark",
                    color_continuous_scale=px.colors.sequential.Rainbow,
                )
                fig.update_traces(marker=dict(line=dict(width=0)))
                fig.write_html(self.outdir + "/" + self.name + "_" + str(i) + ".html")

                if self.publish is not None:
                    publish(
                        self.outdir + "/" + self.name + "_" + str(i) + ".html",
                        self.publish,
                    )

                if self.no_noise_plot:
                    df = df[df["orig_tIdx"] >= 0]

                    fig = px.scatter_3d(
                        df,
                        x="X",
                        y="Y",
                        z="Z",
                        color="tIdx",
                        size="features",
                        hover_data=hover_data,
                        template="plotly_dark",
                        color_continuous_scale=px.colors.sequential.Rainbow,
                    )
                    fig.update_traces(marker=dict(line=dict(width=0)))
                    fig.write_html(
                        self.create_base_output_path() + "_" + str(i) + "_no_noise.html"
                    )

                    if self.publish is not None:
                        publish(
                            self.create_base_output_path()
                            + "_"
                            + str(i)
                            + "_no_noise.html",
                            self.publish,
                        )
        else:
            print(">>>plotting in 2D")  # DEBUG , remove message
            data = {
                "X": coords[:, 0:1].numpy(),
                "Y": coords[:, 1:2].numpy(),
                "tIdx": tidx[:, 0:1].numpy(),
                "features": features[:, 0:1].numpy(),
                "idx": idxs[..., np.newaxis],
            }
            hoverdict = {}
            if hoverfeat is not None:
                for j in range(hoverfeat.shape[1]):
                    hoverdict["f_" + str(j)] = hoverfeat[:, j : j + 1]
                data.update(hoverdict)
            # forget about nidx, just create data frame, shuffle truth colors and plot
            df = pd.DataFrame(
                np.concatenate([data[k] for k in data], axis=1),
                columns=[k for k in data],
            )
            df["orig_tIdx"] = df["tIdx"]
            rdst = np.random.RandomState(1234567890)
            shuffle_truth_colors(df, "tIdx", rdst)
            hover_data = ["orig_tIdx", "idx"] + [k for k in hoverdict.keys()]
            fig = px.scatter(
                df,
                x="X",
                y="Y",
                color="tIdx",
                size="features",
                hover_data=hover_data,
                template="plotly_dark",
                color_continuous_scale=px.colors.sequential.Rainbow,
            )
            fig.update_traces(marker=dict(line=dict(width=0)))
            fig.write_html(self.outdir + "/" + self.name + ".html")
            # publish
            if self.publish is not None:
                publish(self.outdir + "/" + self.name + ".html", self.publish)
            # done


class PlotGraphCondensationEfficiency(_DebugPlotBase):
    def __init__(
        self, accumulate_every: int = 10, externally_triggered=False, **kwargs  # how
    ):
        """
        Inputs:
         - t_energy
         - t_idx
         - graph condensation
         - (opt) is_track -> if given tracks are marked as noise (efficiency with hits only)

        Output:
         - t_energy
        """

        super(PlotGraphCondensationEfficiency, self).__init__(
            externally_triggered=externally_triggered, **kwargs
        )

        self.acc_counter = 0
        self.accumulate_every = accumulate_every

        self.only_accumulate_this_time = False

        accumulate = self.plot_every // accumulate_every + 50

        self.num = CumulativeArray(accumulate, name=self.name + "_num")
        self.den = CumulativeArray(accumulate, name=self.name + "_den")

    def get_config(self):
        config = {
            "accumulate_every": self.accumulate_every
        }  # outdir/publish is explicitly not saved and needs to be set again every time
        base_config = super(PlotGraphCondensationEfficiency, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    # overwrite here
    def call(self, t_energy, t_idx, graph_trans, is_track=None, training=None):

        if not self.check_make_plot([t_energy], training):
            return t_energy

        os.system("mkdir -p " + self.outdir)
        try:
            self.plot(t_energy, t_idx, graph_trans, is_track, training)
        except Exception as e:
            raise e
            # do nothing, don't interrupt training because a debug plot failed

        return t_energy

    def check_make_plot(self, inputs, training=None):
        pre = super(PlotGraphCondensationEfficiency, self).check_make_plot(
            inputs, training
        )

        if self.plot_every <= 0 and not self.externally_triggered:  # nothing
            return pre

        self.only_accumulate_this_time = False
        # OR:
        if self.accumulate_every < self.acc_counter:
            self.acc_counter = 0
            self.only_accumulate_this_time = not pre

            return True

        self.acc_counter += 1
        return pre

    def plot(self, t_energy, t_idx, graph_trans, is_track=None, training=None):
        """
        'rs_down',
            'rs_up',
            'nidx_down',
            'distsq_down', #in case it's needed
            'sel_idx_up',
        """
        rs = graph_trans["rs_down"]
        rsup = graph_trans["rs_up"]

        up_t_idx = tf.gather_nd(t_idx, graph_trans["sel_idx_up"])
        up_t_energy = tf.gather_nd(t_energy, graph_trans["sel_idx_up"])
        up_is_track = (
            tf.gather_nd(is_track, graph_trans["sel_idx_up"])
            if is_track is not None
            else None
        )

        orig_energies = []
        energies = []

        for i in tf.range(rs.shape[0] - 1):

            rs_t_idx = t_idx[rs[i] : rs[i + 1]][:, 0]
            rs_t_energy = t_energy[rs[i] : rs[i + 1]][:, 0]
            rs_is_track = (
                is_track[rs[i] : rs[i + 1]][:, 0] if is_track is not None else None
            )

            if (
                rs_is_track is not None
            ):  # mark tracks as noise so that they are removed from the efficiency calc.
                rs_t_idx = tf.where(rs_is_track > 0, -1, rs_t_idx)

            u, _ = tf.unique(rs_t_energy[rs_t_idx >= 0])

            orig_energies.append(u.numpy())

            rs_sel_t_idx = up_t_idx[rsup[i] : rsup[i + 1]]
            rs_sel_t_energy = up_t_energy[rsup[i] : rsup[i + 1]]

            if rs_is_track is not None:
                rs_up_is_track = up_is_track[rsup[i] : rsup[i + 1]]
                rs_sel_t_idx = tf.where(rs_up_is_track > 0, -1, rs_sel_t_idx)

            # same for selected
            u, _ = tf.unique(rs_sel_t_energy[rs_sel_t_idx >= 0])

            energies.append(u.numpy())

        orig_energies = np.concatenate(orig_energies, axis=0)
        energies = np.concatenate(energies, axis=0)

        bins = np.logspace(-1, 2.3, num=16)  # roughly up to 200

        h, bins = np.histogram(energies, bins=bins)
        h = np.array(h, dtype="float32")

        self.num.put(h)

        h_orig, _ = np.histogram(orig_energies, bins=bins)
        h_orig = np.array(h_orig, dtype="float32")

        self.den.put(h_orig)

        if self.only_accumulate_this_time:
            return

        print(self.name, "plotting...")

        ##interface to old code
        h = self.num.get()
        h_orig = self.den.get()

        h /= h_orig + 1e-3

        h = np.where(h_orig == 0, np.nan, h)

        # make bins points
        bins = bins[:-1] + (bins[1:] - bins[:-1]) / 2.0

        fig = px.line(x=bins, y=h, template="plotly_dark", log_x=True)

        fig.update_layout(
            xaxis_title="Truth shower energy [GeV]",
            yaxis_title="Efficiency",
        )

        fig.write_html(self.outdir + "/" + self.name + ".html")
        if self.publish is not None:
            publish(self.outdir + "/" + self.name + ".html", self.publish)
