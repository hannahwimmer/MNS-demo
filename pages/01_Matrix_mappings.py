import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.cm as cm
import matplotlib.colors as mcolors

st.set_page_config(page_title="Matrix Mapping Demo", layout="wide")
st.title("Matrix mapping")

A = [[0.5,1],[0,-2]]

st.markdown(r"We regard the following matrix $A \in \mathbb{R}^{2\times2}$:")
st.latex(r"""
    A = \begin{pmatrix}
    0.5 & 1 \\
    0 & -2
    \end{pmatrix}
""")

# --- Unit circle vectors ---
theta = np.linspace(0, 2 * np.pi, 100)
unit_vectors = np.vstack((np.cos(theta), np.sin(theta)))   # shape (2,100)
mapped_vectors = A @ unit_vectors                           # shape (2,100)


num_vectors = unit_vectors.shape[1]
unit_colors = [mcolors.to_hex(cm.Grays(i / (num_vectors - 1))) for i in range(num_vectors)]
mapped_colors = [mcolors.to_hex(cm.Reds(i / (num_vectors - 1))) for i in range(num_vectors)]


st.markdown("""What does this matrix **do** to an arbitrary input vector, now? How are 
    vectors **deformed** (i.e., rotated and rescaled) under the transformation corresponding
    to $A$?  
    To understand this, we look at what happens to vectors we understand very
    well, vectors of length 1 that point into any possible room direction: i.e., **vectors
    along the unit circle**!""")

st.markdown("""We draw 100 **evenly-spaced vectors** along the unit circle (gray) - the first
    vector (pointing along the x-axis) is drawn in bright color, the last vector is
    drawn in dark color.""")

st.markdown("""Each of this vectors is now **mapped** with matrix $A$ (red). Again, the 
    mapping of the first unit vector is drawn in bright color, the mapping of the last
    unit vector in dark color (so that we can trace where each vector is mapped to!)""")


cont = st.container(border=True)
with cont:
    st.markdown("""We clearly see that most vectors are mapped to a **completely different direction**.
        Additionally, the usually have a **different length** than they had **prior** to the 
        mapping.""")

    st.badge("""The matrix usually changes both direction and length of the input vectors,
        completely distorting them!""",color='red')

# --- Unit circle vectors ---
theta = np.linspace(0, 2 * np.pi, 100)
unit_vectors = np.vstack((np.cos(theta), np.sin(theta)))   # shape (2,100)
mapped_vectors = A @ unit_vectors                           # shape (2,100)


num_vectors = unit_vectors.shape[1]
unit_colors = [mcolors.to_hex(cm.Grays(i / (num_vectors - 1))) for i in range(num_vectors)]
mapped_colors = [mcolors.to_hex(cm.Reds(i / (num_vectors - 1))) for i in range(num_vectors)]



fig = go.Figure()

fig.add_trace(go.Scatter(
    x=unit_vectors[0], y=unit_vectors[1],
    fill="toself",
    fillcolor="rgba(125,125,125,0.3)",  # purple with opacity
    line=dict(color="gray", width=0.5),
    name="Unit Circle"
))


fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=mapped_vectors[0], y=mapped_vectors[1],
    fill="toself",
    fillcolor="rgba(255,0,0,0.3)",  # mediumseagreen with opacity
    line=dict(color="red", width=0.5),
    name="Transformed Ellipse"
))

# --- Add vectors from origin with smaller arrowheads ---
arrow_step = 1  # fewer arrows for clarity
for i in range(0, 100, arrow_step):
    # Unit circle vectors (purple)
    x, y = unit_vectors[:, i]
    fig.add_annotation(
        x=x, y=y,
        ax=0, ay=0,
        xref="x", yref="y",
        axref="x", ayref="y",
        showarrow=True,
        arrowhead=1,           # smaller arrowhead
        arrowsize=0.75,           # smaller size
        arrowwidth=1.5,
        arrowcolor=unit_colors[i],
        opacity=0.6
    )
    # Mapped vectors (medium green)
    x, y = mapped_vectors[:, i]
    fig2.add_annotation(
        x=x, y=y,
        ax=0, ay=0,
        xref="x", yref="y",
        axref="x", ayref="y",
        showarrow=True,
        arrowhead=1,
        arrowsize=0.75,
        arrowwidth=1.5,
        arrowcolor=mapped_colors[i],
        opacity=0.6
    )

# --- Layout ---
fig.update_layout(
    width=700, height=700,
    xaxis=dict(scaleanchor="y", scaleratio=1, range=[-2, 2], zeroline=True),
    yaxis=dict(range=[-2, 2], zeroline=True),
    template="plotly_white",
    title="Vectors on the unit circle...",
    legend=dict(x=0.02, y=0.98)
)

fig2.update_layout(
    width=700, height=700,
    xaxis=dict(scaleanchor="y", scaleratio=1, range=[-2, 2], zeroline=True),
    yaxis=dict(range=[-2, 2], zeroline=True),
    template="plotly_white",
    title="... mapped via matrix A",
    legend=dict(x=0.02, y=0.98)
)

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig, use_container_width=True)
with col2:
    st.plotly_chart(fig2, use_container_width=True)

st.caption(
    "Gray vectors and area: unit circle (left). "
    "Red vectors and area: image under matrix A (right). "
)
st.markdown("---")
