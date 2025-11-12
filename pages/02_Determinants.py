import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from shapely.geometry import Polygon


st.set_page_config(page_title="Determinant Demo", layout="wide")
st.title("Determinant of a matrix")

A = [[0.8,1],[0,-2]]

st.markdown(r"We regard the following matrix $A \in \mathbb{R}^{2\times2}$:")
st.latex(r"""
\begin{aligned}
A &= \begin{pmatrix}
a_{11} & a_{12} \\
a_{21} & a_{22}
\end{pmatrix}
=
\begin{pmatrix}
0.8 & 1 \\
0 & -2
\end{pmatrix}
\end{aligned}
""")

st.markdown("""The **determinant** of a matrix is a **scalar** value (i.e., $\in \mathbb{R}$)
    that describes how vectors are **mapped** by a quadratic matrix $A$.""")
st.badge("""Important: the determinant is only defined for **quadratic matrices**!""", color='red')

st.markdown("""Let's take a look at the matrix above. We can calculate its determinant
    via:""")
st.latex(r"""
\begin{aligned}
\det(A) 
&= 
\begin{vmatrix}
a_{11} & a_{12} \\
a_{21} & a_{22}
\end{vmatrix}
= a_{11}a_{22} - a_{12}a_{21}
&= 0.8 \cdot (-2) - 1\cdot0
&= -1.6
\end{aligned}
""")

st.markdown("""What does the determinant tell us in detail, though?""")

theta = np.linspace(0, 2 * np.pi, 100)
unit_vectors = np.vstack((np.cos(theta), np.sin(theta)))   # shape (2,100)
mapped_vectors = A @ unit_vectors                           # shape (2,100)


num_vectors = unit_vectors.shape[1]
unit_colors = [mcolors.to_hex(cm.Grays(i / (num_vectors - 1))) for i in range(num_vectors)]
mapped_colors = [mcolors.to_hex(cm.Reds(i / (num_vectors - 1))) for i in range(num_vectors)]

circle_outline = Polygon(zip(unit_vectors[0], unit_vectors[1]))
circle_area = circle_outline.area

ellipse_outline = Polygon(zip(mapped_vectors[0], mapped_vectors[1]))
ellipse_area = ellipse_outline.area
det = np.linalg.det(A)
det_calc = ellipse_area/circle_area

col1, col2 = st.columns(2)
with col1:
    cont = st.container(border=True)
    with cont:
        st.badge("The **value** of the determinant", color='red')
        st.markdown("""The **absolute value** of the determinant tells us how much the **area**
            spanned by the vectors **increased** or **decreased** after compared to before
            the mapping!""")
        st.latex(r"""
            det(A) = \frac{A_{after}}{A_{before}} = \frac{A_{ellipse}}{A_{circle}}
        """)
        st.markdown("""Let's quickly check if this is true for our matrix using the
            shapely.geometry package:""")
        st.latex(fr"""
        \begin{{aligned}}
        &A_{{\text{{circle}}}} = \pi r^2 = \pi(1)^2 = \pi \approx 3.14 \\[4pt]
        &A_{{\text{{ellipse}}}} = {ellipse_area:.2f} \\[6pt]
        &|\det(A)| = {abs(det):.2f} = \frac{{{ellipse_area:.2f}}}{{{circle_area:.2f}}} = {det_calc:.2f}
        \end{{aligned}}
        """)
        st.markdown("""Perfect - we see that the absolute value of the determinant really
            tells us by how much the area changes due to the mapping!""")
        

with col2:
    cont = st.container(border=True)
    with cont:
        st.badge("The **sign** of the determinant", color='red')
        st.markdown("""The **sign** of the determinant tells us whether the **sense of rotation**
            of the vectors is changed during the mapping.""")
        st.markdown("""Take a look at the figures below. The initial vectors on the
            unit circle (left) were ordered **counter-clockwise**; the first vector 
            points along the x-axis, the second is slightly above, and so on.""")
        st.markdown("""Now look at the image of these vectors (right). Suddenly, the 
            ordering is not counter-clockwise anymore, but **clockwise**; the first 
            vector is mapped slightly below the x-axis, the second one below it, and so
            on.""")
        st.markdown("""The **sense of rotation** changes during the mapping with matrix
            $A$!""")
        st.markdown("""The determinant already indicates this change in rotation by its
            **negative sign**; we wouldn't even have to plot it to know!""")
        


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
    xaxis=dict(scaleanchor="y", scaleratio=1, range=[-3, 3], zeroline=True),
    yaxis=dict(range=[-2, 2], zeroline=True),
    template="plotly_white",
    title="Vectors on the unit circle...",
    legend=dict(x=0.02, y=0.98)
)

fig2.update_layout(
    width=700, height=700,
    xaxis=dict(scaleanchor="y", scaleratio=1, range=[-3, 3], zeroline=True),
    yaxis=dict(range=[-2, 2], zeroline=True),
    template="plotly_white",
    title="... mapped via matrix A",
    legend=dict(x=0.02, y=0.98)
)


fig3 = go.Figure()

fig3.add_trace(go.Scatter(
    x=unit_vectors[0], y=unit_vectors[1],
    fill="toself",
    fillcolor="rgba(125,125,125,0.3)",  # purple with opacity
    line=dict(color="gray", width=0.5),
    name="Unit Circle"
))

fig3.add_trace(go.Scatter(
    x=mapped_vectors[0], y=mapped_vectors[1],
    fill="toself",
    fillcolor="rgba(255,0,0,0.3)",  # mediumseagreen with opacity
    line=dict(color="red", width=0.5),
    name="Transformed Ellipse"
))

fig3.add_annotation(
    x=1.5, y=1,  # slightly above the circle
    text=f"Circle area ≈ {circle_area:.2f}",
    showarrow=False,
    font=dict(color="gray", size=14)
)

fig3.add_annotation(
    x=np.mean(mapped_vectors[0]), 
    y=np.max(mapped_vectors[1])+0.5,  # above ellipse
    text=f"Ellipse area ≈ {ellipse_area:.2f}",
    showarrow=False,
    font=dict(color="red", size=14)
)


fig3.update_layout(
    width=700, height=700,
    xaxis=dict(scaleanchor="y", scaleratio=1, range=[-3, 3], zeroline=True),
    yaxis=dict(range=[-1, 1], zeroline=True),
    template="plotly_white",
    title="Unit circle vectors mapped via the matrix (areas only)",
    legend=dict(x=0.02, y=0.98)
)


col1, col2, col3 = st.columns(3)
with col1:
    st.plotly_chart(fig, use_container_width=True)
with col2:
    st.plotly_chart(fig2, use_container_width=True)
with col3:
    st.plotly_chart(fig3, use_container_width=True)

st.caption(
    "Gray vectors and area: unit circle (left). "
    "Red vectors and area: image under matrix A (middle). "
    "Both areas in one figure for comparison (right)."
)
st.markdown("---")




























