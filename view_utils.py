from __future__ import annotations


def grade_label(value) -> str:
    try:
        return f"{int(float(value))}°"
    except Exception:
        return str(value)


def style_clean_axes(fig, grades=None, hide_y=True):
    if grades is not None:
        fig.update_xaxes(
            title=None,
            tickmode="array",
            tickvals=grades,
            ticktext=[grade_label(g) for g in grades],
            showgrid=False,
            zeroline=False,
            ticks="",
        )
    else:
        fig.update_xaxes(
            title=None,
            showgrid=False,
            zeroline=False,
            ticks="",
        )

    fig.update_yaxes(
        title=None,
        showgrid=False,
        zeroline=False,
        showticklabels=not hide_y,
        ticks="",
        rangemode="tozero",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=18, r=18, t=70, b=25),
    )
    return fig
