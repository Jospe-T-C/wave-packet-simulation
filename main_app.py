"""
app.py -- Streamlit frontend for the Wave Packet Simulation
==============================================================
Run with:   streamlit run app.py

Lets you pick a scenario (free particle / harmonic oscillator / barrier
tunneling), adjust the physical parameters with sliders, run the
simulation, then scrub through time to watch |psi(x,t)|^2 evolve.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

from wave_packet_core import (
    build_grid, gaussian_wave_packet,
    potential_free, potential_harmonic, potential_barrier,
    split_step_evolve, expectation_x, sigma_of_t, normalisation,
    sigma_analytical_free, x_mean_analytical_free,
)

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at 8% 8%, rgba(59,130,246,.13), transparent 28%),
        radial-gradient(circle at 92% 12%, rgba(124,58,237,.12), transparent 26%),
        radial-gradient(circle at 70% 92%, rgba(6,182,212,.10), transparent 30%),
        linear-gradient(135deg, #f8fbff 0%, #eef4ff 48%, #f7f3ff 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111c3a 0%, #182b59 48%, #211b4f 100%);
    border-right: 1px solid rgba(255,255,255,.10);
}
[data-testid="stSidebar"] * { color: #eef4ff !important; }
[data-testid="stSidebar"] .stCaption { color: #b9c7e6 !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.15); }
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #38bdf8, #6366f1);
    color: white !important; border: 0; border-radius: 12px;
    font-weight: 700; box-shadow: 0 8px 20px rgba(37,99,235,.28);
}
.hero {
    padding: 1.35rem 1.55rem; border-radius: 20px;
    background: linear-gradient(135deg, rgba(255,255,255,.94), rgba(239,246,255,.82));
    border: 1px solid rgba(96,165,250,.22);
    box-shadow: 0 14px 40px rgba(30,64,175,.10); margin-bottom: 1rem;
}
.hero h1 { margin: 0; color: #172554; letter-spacing: -.03em; }
.hero p { margin: .45rem 0 0; color: #64748b; font-size: 1rem; }
div[data-testid="stMetric"] {
    background: rgba(255,255,255,.88); border: 1px solid rgba(148,163,184,.25);
    border-radius: 16px; padding: .9rem 1rem;
    box-shadow: 0 8px 24px rgba(15,23,42,.06);
}
.stTabs [data-baseweb="tab-list"] {
    gap: 6px; background: rgba(255,255,255,.62); padding: 6px;
    border-radius: 14px; border: 1px solid rgba(148,163,184,.18);
}
.stTabs [data-baseweb="tab"] { border-radius: 10px; padding: .55rem .8rem; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg,#dbeafe,#ede9fe); color:#1d4ed8; }
.block-container { padding-top: 2rem; padding-bottom: 2.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🌊 Wave Packet Simulation</h1>
    <p>Explore Gaussian wave packets, probability density, phase, and time evolution using the Split-Step Fourier Method.</p>
</div>
""", unsafe_allow_html=True)
st.caption("Time-dependent Schrödinger equation solved with the Split-Step Fourier Method  |  Units: ħ = m = 1")

# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
st.sidebar.markdown("## ⚛️ Simulation Controls")
st.sidebar.markdown("### 🎯 Scenario")
scenario = st.sidebar.radio(
    "Choose a potential",
    ["Free particle", "Harmonic oscillator", "Potential barrier (tunneling)"],
)

st.sidebar.markdown("### 🌊 Wave Packet")
x0 = st.sidebar.slider("Initial position x0", -80.0, 0.0, -40.0, step=1.0)
k0 = st.sidebar.slider("Central wavenumber k0", 0.0, 4.0, 1.5, step=0.1)
sigma0 = st.sidebar.slider("Initial width sigma0", 1.0, 10.0, 3.0, step=0.5)

st.sidebar.markdown("### ⏱️ Time Settings")
dt = st.sidebar.slider("Time step dt", 0.01, 0.1, 0.05, step=0.01)
total_time = st.sidebar.slider("Total simulation time", 5.0, 80.0, 40.0, step=5.0)

# Scenario-specific parameters
if scenario == "Harmonic oscillator":
    omega = st.sidebar.slider("Angular frequency omega", 0.05, 0.5, 0.2, step=0.01)
    st.sidebar.caption("Tip: set sigma0 close to sqrt(1/(2*omega)) for a clean coherent state (no spreading).")
elif scenario == "Potential barrier (tunneling)":
    V0 = st.sidebar.slider("Barrier height V0", 0.5, 5.0, 2.3, step=0.1)
    barrier_width = st.sidebar.slider("Barrier width", 1.0, 15.0, 3.0, step=0.5)

run_button = st.sidebar.button("Run simulation", type="primary")
st.sidebar.caption("💡 Adjust the controls, then press **Run simulation** to apply the changes.")

# ----------------------------------------------------------------------
# Simulation (cached so re-runs on the same params are instant)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Running split-step Fourier simulation...")
def run_simulation(scenario, x0, k0, sigma0, dt, total_time, omega=None, V0=None, barrier_width=None):
    x, k, dx = build_grid(N=1024, L=200.0)
    psi0 = gaussian_wave_packet(x, x0, k0, sigma0)
    n_steps = int(total_time / dt)

    if scenario == "Free particle":
        V = potential_free(x)
    elif scenario == "Harmonic oscillator":
        V = potential_harmonic(x, omega)
    else:
        V = potential_barrier(x, V0, barrier_width)

    history = split_step_evolve(psi0, V, k, dt, n_steps)
    times = np.arange(n_steps + 1) * dt
    return x, k, dx, V, history, times


if "sim" not in st.session_state:
    st.session_state.sim = None

if run_button or st.session_state.sim is None:
    kwargs = {}
    if scenario == "Harmonic oscillator":
        kwargs["omega"] = omega
    elif scenario == "Potential barrier (tunneling)":
        kwargs["V0"] = V0
        kwargs["barrier_width"] = barrier_width

    x, k, dx, V, history, times = run_simulation(
        scenario, x0, k0, sigma0, dt, total_time, **kwargs
    )
    st.session_state.sim = dict(
        scenario=scenario, x=x, k=k, dx=dx, V=V, history=history, times=times,
        x0=x0, k0=k0, sigma0=sigma0,
    )

sim = st.session_state.sim

# ----------------------------------------------------------------------
# Main panel
# ----------------------------------------------------------------------
if sim is not None:
    x, dx, V, history, times = sim["x"], sim["dx"], sim["V"], sim["history"], sim["times"]
    n_steps = len(times) - 1

    frame = st.slider("Time", 0, n_steps, 0, format="")
    t_now = times[frame]
    psi_now = history[frame]

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 2D Probability",
        "🏔️ 3D Landscape",
        "🌀 3D Wave Ribbon",
        "▶️ Traveling Wave",
    ])

    # ================================================================
    # TAB 1 -- original 2D view
    # ================================================================
    with tab1:
        col1, col2 = st.columns([2, 1])

        with col1:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            prob = np.abs(psi_now)**2
            show_potential = sim["scenario"] != "Free particle"
            if show_potential:
                Vscaled = V / np.max(V) * np.max(prob) * 2.5 if np.max(V) > 0 else np.zeros_like(V)
                ax.fill_between(x, Vscaled, color="lightgray", step="mid", label="Potential (schematic)")
            ax.plot(x, prob, color="darkblue", lw=2)
            ax.set_xlabel("x")
            ax.set_ylabel(r"$|\psi(x,t)|^2$")
            ax.set_title(f"t = {t_now:.2f}")
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(0, np.max(np.abs(history)**2) * 1.1)
            if show_potential:
                ax.legend(loc="upper right")
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.metric("Normalisation \u222b|\u03c8|\u00b2dx", f"{normalisation(psi_now, dx):.6f}")
            st.metric("\u27e8x\u27e9(t)", f"{expectation_x(psi_now, x, dx):.2f}")
            st.metric("Width \u03c3(t)", f"{sigma_of_t(psi_now, x, dx):.3f}")
            if sim["scenario"] == "Free particle":
                v_g = k0/1.0
                st.caption(f"Group velocity v_g = \u0127k0/m = {v_g:.3f}")
                st.caption(f"Phase velocity v_p = \u0127k0/2m = {v_g/2:.3f}")

        st.divider()

        # --- Diagnostics over full time range ---
        st.subheader("Time evolution")
        dcol1, dcol2 = st.columns(2)

        step_sample = max(1, n_steps // 100)
        idxs = np.arange(0, n_steps + 1, step_sample)
        t_sampled = times[idxs]
        x_mean_num = np.array([expectation_x(history[i], x, dx) for i in idxs])
        sigma_num = np.array([sigma_of_t(history[i], x, dx) for i in idxs])

        with dcol1:
            fig2, ax2 = plt.subplots(figsize=(5, 3.5))
            ax2.plot(t_sampled, sigma_num, 'o-', ms=3, color="green", label="Numerical")
            if sim["scenario"] == "Free particle":
                ax2.plot(times, sigma_analytical_free(sim["sigma0"], times), 'k--', label="Analytical")
            ax2.axvline(t_now, color="red", lw=1, ls=":")
            ax2.set_xlabel("time")
            ax2.set_ylabel("width \u03c3(t)")
            ax2.legend(fontsize=8)
            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

        with dcol2:
            fig3, ax3 = plt.subplots(figsize=(5, 3.5))
            ax3.plot(t_sampled, x_mean_num, 'o-', ms=3, color="blue", label="Numerical")
            if sim["scenario"] == "Free particle":
                ax3.plot(times, x_mean_analytical_free(sim["x0"], sim["k0"], times), 'k--', label="Analytical")
            ax3.axvline(t_now, color="red", lw=1, ls=":")
            ax3.set_xlabel("time")
            ax3.set_ylabel(r"$\langle x \rangle(t)$")
            ax3.legend(fontsize=8)
            fig3.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)

        st.caption("Drag the Time slider above to scrub through the simulation. Change parameters in the sidebar and click **Run simulation** to recompute.")

    # ================================================================
    # TAB 2 -- 3D probability landscape: |psi(x,t)|^2 as a surface
    #          over the (x, t) plane. This is the most intuitive 3D
    #          view: a "hill" that slides and flattens (free particle),
    #          rocks back and forth (harmonic), or splits at a wall
    #          (barrier).
    # ================================================================
    with tab2:
        st.markdown(
            "**Drag to rotate, scroll to zoom.** Each slice along the *time* "
            "axis is the probability density `|psi(x,t)|^2` at that instant "
            "-- stack every instant together and you get this landscape. "
            "Watch how the ridge moves and spreads (or oscillates, or splits)."
        )

        # Downsample so Plotly stays responsive: aim for roughly
        # 160 x-points and up to 160 t-points.
        x_step = max(1, len(x) // 160)
        t_step = max(1, (n_steps + 1) // 160)
        x_ds = x[::x_step]
        t_idx = np.arange(0, n_steps + 1, t_step)
        t_ds = times[t_idx]
        Z = np.abs(history[np.ix_(t_idx, np.arange(0, len(x), x_step))])**2

        fig3d = go.Figure(data=[go.Surface(
            x=x_ds, y=t_ds, z=Z,
            colorscale="Viridis",
            colorbar=dict(title="|\u03c8|\u00b2"),
        )])

        # Highlight the current time slice (from the Time slider above)
        # as a red curve resting on the surface, so this tab stays in
        # sync with the rest of the app.
        prob_now_ds = np.abs(psi_now[::x_step])**2
        fig3d.add_trace(go.Scatter3d(
            x=x_ds, y=np.full_like(x_ds, t_now), z=prob_now_ds,
            mode="lines", line=dict(color="red", width=6),
            name=f"t = {t_now:.2f}",
        ))

        fig3d.update_layout(
            scene=dict(
                xaxis_title="position x",
                yaxis_title="time t",
                zaxis_title="|\u03c8(x,t)|\u00b2",
                camera=dict(eye=dict(x=1.6, y=-1.6, z=1.0)),
            ),
            height=650,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig3d, use_container_width=True)

    # ================================================================
    # TAB 3 -- 3D wave ribbon: Re(psi) and Im(psi) plotted as two
    #          perpendicular axes vs x, at the CURRENT time from the
    #          slider. This is the classic "spring" / "corkscrew" way
    #          to visualise a complex-valued wave -- watching it twist
    #          as t increases shows the wave's phase evolving, which a
    #          flat |psi|^2 plot completely hides.
    # ================================================================
    with tab3:
        st.markdown(
            f"**Time = {t_now:.2f}** (set with the Time slider in the first tab). "
            "The wavefunction `psi(x,t)` is complex, so it can't be drawn as a "
            "single curve on paper. Here the real part is the *y*-axis and the "
            "imaginary part is the *z*-axis -- so the curve corkscrews along x, "
            "and the corkscrew's radius at each point is exactly `|psi(x,t)|`, "
            "the quantity that gives the probability density. Drag to rotate; "
            "looking straight down the x-axis you'll see a spiral, and looking "
            "from the side you'll see the familiar Re(psi) wiggle."
        )

        x_step_r = max(1, len(x) // 400)
        x_r = x[::x_step_r]
        psi_r = psi_now[::x_step_r]
        re_r = np.real(psi_r)
        im_r = np.imag(psi_r)
        prob_r = np.abs(psi_r)**2

        fig_ribbon = go.Figure(data=[go.Scatter3d(
            x=x_r, y=re_r, z=im_r,
            mode="lines",
            line=dict(
                color=prob_r, colorscale="Plasma", width=8,
                colorbar=dict(title="|\u03c8|\u00b2"),
            ),
        )])

        # A faint projection onto the x-Re(psi) plane (z=0) for orientation
        fig_ribbon.add_trace(go.Scatter3d(
            x=x_r, y=re_r, z=np.zeros_like(x_r),
            mode="lines", line=dict(color="lightgray", width=2),
            showlegend=False,
        ))

        fig_ribbon.update_layout(
            scene=dict(
                xaxis_title="position x",
                yaxis_title="Re(\u03c8)",
                zaxis_title="Im(\u03c8)",
                camera=dict(eye=dict(x=1.4, y=1.4, z=0.6)),
            ),
            height=650,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_ribbon, use_container_width=True)

    # ================================================================
    # TAB 4 -- Real-time animated traveling wave. Two curves animate
    #          together: the oscillating carrier wave Re(psi(x,t)) and
    #          its envelope |psi(x,t)|^2, exactly like the textbook
    #          picture of a "wave packet" (fast ripples inside a slower
    #          moving/spreading envelope). Uses Plotly's native
    #          animation frames + Play button for smooth, continuous
    #          real-time motion (no manual dragging needed).
    # ================================================================
    with tab4:
        st.markdown(
            "Press **Play** below to watch the wave move in real time. "
            "The rippling blue curve is the oscillating part of the "
            "wavefunction, `Re[psi(x,t)]` -- this is the actual *traveling "
            "wave*. The shaded orange curve is the probability envelope "
            "`|psi(x,t)|^2` riding on top of it, showing where the particle "
            "is likely to be found."
        )

        n_frames_target = 150
        f_step = max(1, (n_steps + 1) // n_frames_target)
        f_idx = np.arange(0, n_steps + 1, f_step)

        x_step_a = max(1, len(x) // 500)
        x_a = x[::x_step_a]

        # Scale Re(psi) up so it's visible on the same axis as |psi|^2
        prob_all = np.abs(history[:, ::x_step_a])**2
        re_all = np.real(history[:, ::x_step_a])
        scale = np.max(prob_all) / (np.max(np.abs(re_all)) + 1e-12) * 0.9

        re0 = re_all[f_idx[0]] * scale
        prob0 = prob_all[f_idx[0]]

        fig_anim = go.Figure(
            data=[
                go.Scatter(x=x_a, y=prob0, mode="lines", name="|\u03c8|\u00b2 (envelope)",
                           fill="tozeroy", line=dict(color="orange", width=1)),
                go.Scatter(x=x_a, y=re0, mode="lines", name="Re(\u03c8) (traveling wave)",
                           line=dict(color="royalblue", width=2)),
            ],
            frames=[
                go.Frame(
                    data=[
                        go.Scatter(x=x_a, y=prob_all[i]),
                        go.Scatter(x=x_a, y=re_all[i] * scale),
                    ],
                    name=str(i),
                )
                for i in f_idx
            ],
        )

        y_max = np.max(prob_all) * 1.2
        fig_anim.update_layout(
            xaxis=dict(title="position x", range=[x_a.min(), x_a.max()]),
            yaxis=dict(title="amplitude (arb. units)", range=[-y_max*0.6, y_max]),
            height=500,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=1.15, x=0.0, xanchor="left",
                buttons=[
                    dict(label="\u25b6 Play", method="animate",
                         args=[None, {"frame": {"duration": 40, "redraw": True},
                                       "fromcurrent": True, "transition": {"duration": 0}}]),
                    dict(label="\u23f8 Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
                ],
            )],
            sliders=[dict(
                steps=[dict(method="animate", label=f"{times[i]:.1f}",
                             args=[[str(i)], {"frame": {"duration": 0, "redraw": True},
                                               "mode": "immediate"}])
                       for i in f_idx],
                x=0.0, y=-0.05, len=1.0,
                currentvalue=dict(prefix="t = "),
            )],
        )

        st.plotly_chart(fig_anim, use_container_width=True)
        st.caption(
            "You can also drag the slider under the plot to scrub manually, "
            "or use the Play/Pause buttons above it."
        )
