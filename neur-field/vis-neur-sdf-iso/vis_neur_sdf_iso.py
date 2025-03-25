import argparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import torch
import os

def plot_cuts_iso(neur_sdf_func,
                  box_size=(2.5, 2.5, 2.5), max_n_eval_pts=1e6,
                  resolution=256, thres=0.0, imgs_per_cut=1, save_path=None, device='cuda') -> go.Figure:
    """ plot levelset at a certain cross section, assume inputs are centered
    Args:
        neur_sdf_func: A function to extract the SDF/occupancy logits of (N, 3) points
        box_size (List[float]): bounding box dimension
        max_n_eval_pts (int): max number of points to evaluate in one inference
        resolution (int): cross section resolution xy
        thres (float): levelset value
        imgs_per_cut (int): number of images for each cut (plotted in rows)
    Returns:
        a numpy array for the image
    """
    xmax, ymax, zmax = [b / 2 for b in box_size]
    xx, yy = np.meshgrid(np.linspace(-xmax, xmax, resolution),
                         np.linspace(-ymax, ymax, resolution))
    xx = xx.ravel()
    yy = yy.ravel()

    fig = make_subplots(rows=imgs_per_cut, cols=3,
                        subplot_titles=('xz', 'xy', 'yz'),
                        shared_xaxes='all', shared_yaxes='all',
                        vertical_spacing=0.01, horizontal_spacing=0.01,
                        )

    def _plot_cut(fig, idx, pos, neur_sdf_func, xmax, ymax, resolution):
        """ plot one cross section pos (3, N) """

        # evaluate points in serial
        field_input = torch.tensor(pos.T, dtype=torch.float).to(device)
        feat = torch.zeros((field_input.shape[0], 1)).to(device)
        feat[:, 0] = 1
        values = neur_sdf_func(field_input).flatten()
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
            # np.save('xy.npy', values)
        values = values.reshape(resolution, resolution)
        contour_dict = dict(autocontour=False,
                            colorscale='RdBu',
                            reversescale=True,
                            contours=dict(
                                start=-0.1 + thres,
                                end=0.1 + thres,
                                size=0.01,
                                showlabels=True,  # show labels on contours
                                # labelfont=dict(  # label font properties
                                #     size=12,
                                #     color='white',
                                # )
                            ), )
        r_idx = idx // 3

        fig.add_trace(
            go.Contour(x=np.linspace(-xmax, xmax, resolution),
                       y=np.linspace(-ymax, ymax, resolution),
                       z=values,
                       **contour_dict
                       ),
            col=idx % 3 + 1, row=r_idx + 1  # 1-idx
        )

        fig.update_xaxes(
            range=[-xmax, xmax],  # sets the range of xaxis
            constrain="range",  # meanwhile compresses the xaxis by decreasing its "domain"
            col=idx % 3 + 1, row=r_idx + 1)
        fig.update_yaxes(
            range=[-ymax, ymax],
            col=idx % 3 + 1, row=r_idx + 1
        )

    steps = np.stack([np.linspace(-b / 2, b / 2, imgs_per_cut + 2)[1:-1] for b in box_size], axis=-1)
    for index in range(imgs_per_cut):
        position_cut = [np.vstack([xx, np.full(xx.shape[0], steps[index, 1]), yy]),
                        np.vstack([xx, yy, np.full(xx.shape[0], steps[index, 2])]),
                        np.vstack([np.full(xx.shape[0], steps[index, 0]), xx, yy]), ]
        _plot_cut(
            fig, index * 3, position_cut[0], neur_sdf_func, xmax, zmax, resolution)
        _plot_cut(
            fig, index * 3 + 1, position_cut[1], neur_sdf_func, xmax, ymax, resolution)
        _plot_cut(
            fig, index * 3 + 2, position_cut[2], neur_sdf_func, ymax, zmax, resolution)

    fig.update_layout(
        title='iso-surface',
        height=1200 * imgs_per_cut,
        width=1200 * 3,
        autosize=False,
        scene=dict(aspectratio=dict(x=1, y=1))
    )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
        print("HTML file is saved to:", save_path)

    return fig

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser(
        description="Visualize the iso-surfaces of a 3D neural signed distance field."
    )
    parser.add_argument("-i", "--input", required=True, help="Input file path, a network model file in pytorch(*.pt) that takes a set of 3D coordinates as input and outputs the corresponding implicit values or signed distances.")
    parser.add_argument("-o", "--output", required=True, help="Output file path, a html file that can be opened in a browser.")

    parser.add_argument("--cpu", action="store_true", help="cpu mode")

    args = parser.parse_args()

    device = 'cpu' if args.cpu or not torch.cuda.is_available() else 'cuda'

    model_path = args.input
    html_path = os.path.join(os.getcwd(), args.output)

    # load model
    try:
        neur_sdf_func = torch.jit.load(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print("Error loading model:", e)

    neur_sdf_func.to(device)
    neur_sdf_func.eval()
    with torch.no_grad():
        plot_cuts_iso(neur_sdf_func, save_path=html_path, device=device)

