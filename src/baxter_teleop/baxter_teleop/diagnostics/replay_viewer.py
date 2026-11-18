"""Compare a recorded episode with its replay: cameras, TCP position and orientation.

    python -m baxter_teleop.diagnostics.replay_viewer EPISODE [--port 8051]

EPISODE is a session episode directory (e.g. data/sessions/BlueCupRed/episode_001)
that contains a replay/ subdirectory with the replay bag.

Row 1: camera frames from the recording, scrubbed by a time slider.
Row 2: two plots side by side. Left: TCP position (x, y, z), recording vs replay.
       Right: TCP orientation as RPY (degrees), recording vs replay.
"""

import argparse
import base64
from pathlib import Path

import numpy as np

TOPIC_TCP = '/teleop/left/tcp'
TOPIC_ENDPOINT = '/robot/limb/left/endpoint_state'
IMAGE_TOPICS = [
    '/cameras/left_hand_camera/image/processed/compressed',
    '/cameras/right_hand_camera/image/processed/compressed',
    '/cameras/third_eye/color/image_raw/compressed',
]
IMAGE_LABELS = ['Left hand', 'Right hand', 'Third eye']


def _find_bags(episode):
    recording = Path(episode)
    replay = recording / 'replay'
    if not replay.is_dir():
        raise SystemExit(f'{replay} not found: run replay_demo.sh --bag first')
    infer_dirs = sorted(replay.glob('infer_*'))
    inference = infer_dirs[0] if infer_dirs else None
    return recording, replay, inference


def _pick_tcp_topic(rec):
    for t in [TOPIC_TCP, TOPIC_ENDPOINT]:
        if any(ti.name == t for ti in rec.topics):
            return t
    raise SystemExit(f'no TCP topic in {rec.path}')


def _load_tcp(rec, topic):
    df = rec.table(topic)
    t_ns = df['_header_stamp_ns'].to_numpy().astype(np.int64)
    t_s = (t_ns - t_ns[0]) / 1e9
    prefix = 'pose.' if 'pose.position.x' in df.columns else ''
    xyz = np.column_stack([
        df[f'{prefix}position.x'].to_numpy(),
        df[f'{prefix}position.y'].to_numpy(),
        df[f'{prefix}position.z'].to_numpy(),
    ])
    qx = df[f'{prefix}orientation.x'].to_numpy()
    qy = df[f'{prefix}orientation.y'].to_numpy()
    qz = df[f'{prefix}orientation.z'].to_numpy()
    qw = df[f'{prefix}orientation.w'].to_numpy()
    roll = np.degrees(np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2)))
    pitch = np.degrees(np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1)))
    yaw = np.degrees(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2)))
    rpy = np.column_stack([roll, pitch, yaw])
    return t_s, xyz, rpy


def _align_offset(rec_t, rec_x, rep_t, rep_x):
    """Find the time offset (s) to subtract from the replay so it aligns with the recording."""
    dt = 0.01
    t_grid = np.arange(0, rec_t[-1], dt)
    rec_xi = np.interp(t_grid, rec_t, rec_x)
    best_offset, best_rms = 0.0, 1e9
    for offset_ms in range(0, 8000, 10):
        offset = offset_ms / 1000.0
        t_shifted = t_grid + offset
        mask = t_shifted <= rep_t[-1]
        if mask.sum() < 100:
            continue
        rep_xi = np.interp(t_shifted[mask], rep_t, rep_x)
        rms = np.sqrt(np.mean((rec_xi[mask] - rep_xi) ** 2))
        if rms < best_rms:
            best_rms = rms
            best_offset = offset
    return best_offset


def _available_image_topics(rec):
    topic_names = {t.name for t in rec.topics}
    return [(topic, label) for topic, label in zip(IMAGE_TOPICS, IMAGE_LABELS) if topic in topic_names]


def serve(episode, port=8051, host='127.0.0.1'):
    import plotly.graph_objects as go
    from dash import Dash, Input, Output, State, dcc, html, no_update
    from plotly.subplots import make_subplots
    from recap.recording import Recording

    recording_path, replay_path, inference_path = _find_bags(episode)
    rec_recording = Recording(recording_path)
    rec_replay = Recording(replay_path)
    rec_inference = Recording(inference_path) if inference_path else None

    rec_topic = _pick_tcp_topic(rec_recording)
    rep_topic = _pick_tcp_topic(rec_replay)
    rec_t, rec_xyz, rec_rpy = _load_tcp(rec_recording, rec_topic)
    rep_t, rep_xyz, rep_rpy = _load_tcp(rec_replay, rep_topic)

    offset = _align_offset(rec_t, rec_xyz[:, 0], rep_t, rep_xyz[:, 0])
    rep_t = rep_t - offset
    print(f'Time alignment: replay shifted by {offset:.2f} s')

    has_inference = rec_inference is not None
    if has_inference:
        inf_topic = _pick_tcp_topic(rec_inference)
        inf_t, inf_xyz, inf_rpy = _load_tcp(rec_inference, inf_topic)
        inf_offset = _align_offset(rec_t, rec_xyz[:, 0], inf_t, inf_xyz[:, 0])
        inf_t = inf_t - inf_offset
        print(f'Time alignment: inference shifted by {inf_offset:.2f} s')
    else:
        inf_t = inf_xyz = inf_rpy = None
        print('No inference bag found')

    rec_image_topics = _available_image_topics(rec_recording)
    rep_image_topics = _available_image_topics(rec_replay)
    inf_image_topics = _available_image_topics(rec_inference) if has_inference else []
    image_topics = rec_image_topics
    duration = rec_t[-1]

    # Load initial camera frames (opens the reader, which stays open for the server)
    initial_frames = []
    for topic, _ in image_topics:
        _, data, mime = rec_recording.frame_encoded(topic, rec_recording.start_ns)
        initial_frames.append(f'data:{mime};base64,{base64.b64encode(data).decode()}')

    ink = '#0b0b0b'
    ink2 = '#52514e'
    grid = '#e4e3df'
    rec_color = '#0b0b0b'
    rep_color = '#2a78d6'
    inf_color = '#d64a2a'

    axis_style = dict(
        showgrid=True,
        gridcolor=grid,
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor=ink2,
        linewidth=1,
        ticks='outside',
        tickcolor=ink2,
        ticklen=4,
    )
    layout_common = dict(
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family='sans-serif', size=11, color=ink),
        legend=dict(orientation='h', x=0.5, xanchor='center', y=1.0, yanchor='bottom', bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=64, r=16, t=28, b=44),
        hovermode='x unified',
    )

    def _make_fig(labels_units, rec_data, rep_data, i_data=None):
        fig = make_subplots(rows=len(labels_units), cols=1, shared_xaxes=True, vertical_spacing=0.1)
        for i, (label, unit) in enumerate(labels_units):
            row = i + 1
            fig.add_trace(
                go.Scatter(
                    x=rec_t,
                    y=rec_data[:, i],
                    name='recording',
                    line=dict(color=rec_color, width=1.5),
                    legendgroup='rec',
                    showlegend=(i == 0),
                ),
                row=row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=rep_t,
                    y=rep_data[:, i],
                    name='replay',
                    line=dict(color=rep_color, width=1.5),
                    legendgroup='rep',
                    showlegend=(i == 0),
                ),
                row=row,
                col=1,
            )
            if i_data is not None:
                fig.add_trace(
                    go.Scatter(
                        x=inf_t,
                        y=i_data[:, i],
                        name='inference',
                        line=dict(color=inf_color, width=1.5),
                        legendgroup='inf',
                        showlegend=(i == 0),
                    ),
                    row=row,
                    col=1,
                )
            fig.update_yaxes(
                title_text=f'{label} ({unit})', title_font_size=10, title_standoff=8, row=row, col=1, **axis_style
            )
        fig.update_xaxes(title_text='time (s)', row=len(labels_units), col=1, **axis_style)
        fig.update_layout(height=480, **layout_common)
        return fig

    pos_fig = _make_fig([('x', 'm'), ('y', 'm'), ('z', 'm')], rec_xyz, rep_xyz, inf_xyz)
    rpy_fig = _make_fig([('roll', 'deg'), ('pitch', 'deg'), ('yaw', 'deg')], rec_rpy, rep_rpy, inf_rpy)

    app = Dash(__name__)
    app.title = f'Replay: {recording_path.parent.name}/{recording_path.name}'
    app.index_string = """<!DOCTYPE html>
<html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
.rc-slider-track { background-color: #333 !important; }
.rc-slider-handle, .rc-slider-handle:hover, .rc-slider-handle:active, .rc-slider-handle:focus {
    border-color: #333 !important; background-color: #333 !important;
    box-shadow: none !important; opacity: 1 !important;
}
.rc-slider-dot-active { border-color: #333 !important; }
[class*="slider"] [role="slider"] {
    border-color: #333 !important; background-color: #333 !important;
}
input[type="radio"] { accent-color: #333; }
</style>
</head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>"""

    slider_marks = {i: f'{i}s' for i in range(0, int(duration) + 1, max(1, int(duration) // 10))}

    camera_imgs = [
        html.Div(
            [
                html.Div(
                    label,
                    style={
                        'textAlign': 'center',
                        'fontSize': '11px',
                        'fontWeight': '600',
                        'color': ink2,
                        'marginBottom': '2px',
                    },
                ),
                html.Img(
                    id=f'cam-{i}',
                    src=initial_frames[i] if i < len(initial_frames) else '',
                    style={
                        'width': '100%',
                        'maxHeight': '280px',
                        'objectFit': 'contain',
                        'display': 'block',
                        'background': '#222',
                    },
                ),
            ],
            style={'flex': '1', 'minWidth': '0'},
        )
        for i, (_, label) in enumerate(image_topics)
    ]

    has_replay_cams = len(rep_image_topics) > 0
    has_inference_cams = len(inf_image_topics) > 0
    step = 0.04

    heading_style = {
        'fontSize': '13px',
        'fontWeight': '600',
        'color': ink,
        'textAlign': 'center',
        'padding': '8px 0 0 0',
    }

    btn_style = {
        'border': f'1px solid {ink2}',
        'background': 'white',
        'color': ink,
        'padding': '2px 12px',
        'fontSize': '13px',
        'cursor': 'pointer',
        'fontFamily': 'monospace',
        'lineHeight': '1.6',
    }

    app.layout = html.Div(
        [
            dcc.Interval(id='play-interval', interval=40, disabled=True),
            # Row 1: cameras + controls
            html.Div(
                [
                    html.Div(camera_imgs, style={'display': 'flex', 'gap': '8px'}),
                    html.Div(
                        [
                            html.Button('Play', id='play-btn', n_clicks=0, style=btn_style),
                            dcc.RadioItems(
                                id='cam-source',
                                options=[
                                    {'label': ' Recording', 'value': 'rec'},
                                    {'label': ' Replay', 'value': 'rep', 'disabled': not has_replay_cams},
                                    {'label': ' Inference', 'value': 'inf', 'disabled': not has_inference_cams},
                                ],
                                value='rec',
                                inline=True,
                                style={
                                    'fontSize': '12px',
                                    'display': 'inline-block',
                                    'marginLeft': '16px',
                                    'verticalAlign': 'middle',
                                },
                                inputStyle={'marginRight': '4px'},
                                labelStyle={'marginRight': '16px'},
                            ),
                        ],
                        style={'padding': '6px 0', 'display': 'flex', 'alignItems': 'center'},
                    ),
                    dcc.Slider(
                        id='time-slider',
                        min=0,
                        max=round(duration, 1),
                        step=step,
                        value=0,
                        marks=slider_marks,
                        tooltip={'placement': 'bottom', 'always_visible': True},
                        updatemode='drag',
                    ),
                ],
                style={'padding': '8px 12px'},
            ),
            # Row 2: two plots side by side
            html.Div(
                [
                    html.Div(
                        [
                            html.Div('TCP position', style=heading_style),
                            dcc.Graph(
                                id='pos-plot', figure=pos_fig, config={'scrollZoom': True, 'displayModeBar': False}
                            ),
                        ],
                        style={'flex': '1', 'minWidth': '0'},
                    ),
                    html.Div(
                        [
                            html.Div('TCP orientation', style=heading_style),
                            dcc.Graph(
                                id='rpy-plot', figure=rpy_fig, config={'scrollZoom': True, 'displayModeBar': False}
                            ),
                        ],
                        style={'flex': '1', 'minWidth': '0'},
                    ),
                ],
                style={'display': 'flex', 'gap': '16px', 'padding': '0 8px'},
            ),
        ],
        style={'fontFamily': 'sans-serif', 'maxWidth': '1400px', 'margin': '0 auto', 'padding': '16px 24px'},
    )

    @app.callback(
        Output('play-interval', 'disabled'),
        Output('play-btn', 'children'),
        Input('play-btn', 'n_clicks'),
        State('play-interval', 'disabled'),
    )
    def toggle_play(n, disabled):
        if n == 0:
            return True, 'Play'
        playing = disabled
        return not playing, 'Pause' if playing else 'Play'

    @app.callback(
        Output('time-slider', 'value'),
        Input('play-interval', 'n_intervals'),
        State('time-slider', 'value'),
        State('play-interval', 'disabled'),
    )
    def advance_slider(n, current, disabled):
        if disabled or n is None:
            return no_update
        nxt = round(current + step, 2)
        if nxt > duration:
            return 0
        return nxt

    if image_topics:

        @app.callback(
            [Output(f'cam-{i}', 'src') for i in range(len(image_topics))],
            Input('time-slider', 'value'),
            Input('cam-source', 'value'),
        )
        def update_cameras(t, source):
            t_s = float(t)
            if source == 'inf' and has_inference_cams:
                bag = rec_inference
                topics = inf_image_topics
                t_ns = bag.start_ns + int((t_s + inf_offset) * 1e9)
            elif source == 'rep' and has_replay_cams:
                bag = rec_replay
                topics = rep_image_topics
                t_ns = bag.start_ns + int((t_s + offset) * 1e9)
            else:
                bag = rec_recording
                topics = rec_image_topics
                t_ns = bag.start_ns + int(t_s * 1e9)
            results = []
            topic_map = {tp: lb for tp, lb in topics}
            for rec_topic, _ in image_topics:
                if rec_topic in topic_map:
                    _, data, mime = bag.frame_encoded(rec_topic, t_ns)
                    results.append(f'data:{mime};base64,{base64.b64encode(data).decode()}')
                else:
                    results.append('')
            return results

    print(f'Replay viewer: http://{host}:{port}')
    print(f'  recording:  {recording_path}')
    print(f'  replay:     {replay_path}')
    if has_inference:
        print(f'  inference:  {inference_path}')
    print(f'  duration:   {duration:.1f} s, {len(image_topics)} cameras')
    app.run(host=host, port=port, debug=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('episode', help='episode directory with a replay/ subdirectory')
    parser.add_argument('--port', type=int, default=8051)
    parser.add_argument('--host', default='127.0.0.1')
    args = parser.parse_args()
    serve(args.episode, args.port, args.host)


if __name__ == '__main__':
    main()
