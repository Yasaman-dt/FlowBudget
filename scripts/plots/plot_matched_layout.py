"""Export two-panel figures using the FID-versus-NFE PNG geometry."""


def save_matched_png(fig, axes, path, top_margin=100):
    dpi = 300
    width, height = 1959, 930
    fig.set_size_inches(width / dpi, height / dpi)
    for ax, left in zip(axes, (189, 1143)):
        ax.set_box_aspect(643 / 804)
        ax.set_position([left / width, (height - top_margin - 643) / height,
                         804 / width, 643 / height])
    fig.savefig(path, dpi=dpi, bbox_inches=None, facecolor='white')
