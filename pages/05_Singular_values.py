import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.cm as cm
import matplotlib.colors as mcolors

st.set_page_config(page_title="Singular Values Demo", layout="wide")
st.title("Singular Values and the SVD")

st.markdown(r"""
We now turn from **eigenvalues and eigenvectors** to **singular values** and the **singular value decomposition (SVD)**.
While eigenvalues apply mainly to *square* matrices, singular values are defined for **any** matrix 
$A \in \mathbb{R}^{m \times n}$ and provide similar geometric insight.  
While, with the eigenvector discussion, we want to find vectors that are parallel or
antiparallel to themselves after mappen (in the most abstracted sense), we now, with
the singular vector discussion, want to find a set of vectors that are pairwise 
**orthogonal** to each other before **and** after mapping!
""")

cont = st.container(border=True)
with cont:
    st.badge("A short motivation", color='red')
    st.markdown(r"""We want to find the maximum possible set of vectors in our input 
        space that are **orthogonal** to each other and **remain** orthogonal even after
        mapping with matrix $A$. In math:""")
    st.latex(r"""
        Av_1 = w_1 = \sigma_1 u_1  \\
        Av_2 = w_2 = \sigma_2 u_2
    """)
    st.markdown(r"""with $v_1 \perp v_2$ and $w_1 \perp w_2$ resp. $u_1 \perp u_2$. 
        As we know the matrix might stretch or compress our vectors, we just 'pulled out'
        the factor in the last part, leaving $u_i$ to be normed just like $v_i$.  
        We can now write this more compactly in **matrix notation**:""")
    st.latex(r"""
        A\left[ v_1, v_2 \right] = AV = \left[u_1, u_2\right] 
            \begin{pmatrix}
                \sigma_1 & 0 \\
                0 & \sigma_2
            \end{pmatrix} = U\Sigma""")
    st.markdown(r"""Now as the vectors $v_i$ are orthogonal to each other, $V^T V = \mathbb{I}$,
    meaning the transpose corresponds to the inverse ($V^T = V^{-1}$) - the same holds 
    for $U$ as well.  
    What we then get is a representation of the matrix $A$ via **three different matrices**:""")
    st.latex(r"""
        A = U\Sigma V^T    
    """)
    st.markdown(r"""The equation just above is called the **singular value decomposition**
        (SVD) of matrix $A$.  
        In contrast to the diagonalization via the eigenvalues of a matrix,
        the SVD does **not** require the matrix to be quadratic - **any** arbitrary matrix
        has an SVD.""")


cont2 = st.container(border=True)
with cont2:
    st.badge("Understanding the Three Matrices", color="red")
    st.markdown(r"""
        Each of the three matrices in the decomposition
        $$A = U \Sigma V^T$$
        has a **distinct geometric role** in the mapping process:
        """)

    st.markdown(r"""
        - **$V^T$**: rotates (or reflects) the **input space** so that the new coordinate 
          axes align with the *principal directions* of the mapping.  
          These directions are the **right singular vectors**.
        - **$\Sigma$**: stretches or compresses along the coordinate axes by the
          singular values $\sigma_i$.
          This step turns the unit circle into an **ellipse** whose axes lengths are the 
          **singular values**.
        - **$U$**: finally rotates (or reflects) the resulting ellipse in the **output space**, 
          aligning it with the image of the transformation.
        """)

    st.latex(r"""
        \text{input vector } x
        \xrightarrow{\;V^T\;}
        \text{rotated to principal coordinates}
        \xrightarrow{\;\Sigma\;}
        \text{scaled}
        \xrightarrow{\;U\;}
        \text{rotated into output space.}
        """)
    st.markdown(r"""**In short**: the mapping of matrix $A$, complex as it may be, can be 
        split into a sequence of easy-to-understand matrices $U$, $\Sigma$, and $V^T$.""")
    
number_vectors = 100
room_angle = np.linspace(0, 2*np.pi, number_vectors)
initial_vectors = np.array([np.cos(room_angle), np.sin(room_angle)])

A = np.array([[0.8, 1],
              [0, -2]])

mapped_vectors = A @ initial_vectors
U, S, Vt = np.linalg.svd(A)
Vtx = Vt @ initial_vectors
SigmaVtx = np.diag(S) @ Vtx
USigmaVtx = U @ SigmaVtx


col1, col2, col3, col4 = st.columns(4, border=True)
num_vectors = initial_vectors.shape[1]
unit_colors = [mcolors.to_hex(cm.Grays(i / (num_vectors - 1))) for i in range(num_vectors)]
mapped_colors = [mcolors.to_hex(cm.Reds(i / (num_vectors - 1))) for i in range(num_vectors)]

V_colors = [mcolors.to_hex(cm.GnBu(i / (num_vectors - 1))) for i in range(num_vectors)]
Sigma_colors = [mcolors.to_hex(cm.RdPu(i / (num_vectors - 1))) for i in range(num_vectors)]
U_colors = [mcolors.to_hex(cm.OrRd(i / (num_vectors - 1))) for i in range(num_vectors)]


with col1:

    fig = go.Figure()

    # --- Layout ---
    fig.update_layout(
        width=700,
        height=700,
        xaxis=dict(scaleanchor="y", scaleratio=1, range=[-2, 2], zeroline=True),
        yaxis=dict(range=[-2, 2], zeroline=True),
        template="plotly_white",
        title="Mapping via matrix A",
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
    )

    # --- Add the unit circle polygon ---
    fig.add_trace(go.Scatter(
        x=initial_vectors[0],
        y=initial_vectors[1],
        fill="toself",
        mode="lines",
        line=dict(color='rgba(125,125,125,0.8)', width=1),
        fillcolor='rgba(125,125,125,0.2)',
        opacity=0.5,
        name="Unit Circle"
    ))

    fig.add_trace(go.Scatter(
        x=mapped_vectors[0],
        y=mapped_vectors[1],
        fill="toself",
        mode="lines",
        line=dict(color='rgba(255,0,0,0.8)', width=1),
        fillcolor='rgba(255,0,0,0.2)',
        opacity=0.5,
        name="Unit circle mapped via A"
    ))

    # --- Draw arrows (vectors) ---
    arrow_step = max(1, initial_vectors.shape[1] // 30)  # ~30 arrows for clarity

    for i in range(0, initial_vectors.shape[1], arrow_step):
        # Unit circle vectors (purple)
        x0, y0 = 0, 0
        x1, y1 = initial_vectors[:, i]
        fig.add_scatter(x=np.array([0, x1]),
                        y=np.array([0, y1]),
                        mode='lines',
                        line=dict(color=unit_colors[i], width=1),
                        showlegend=False
                        )

        fig.add_annotation(
            x=x1, y=y1,
            ax=x0, ay=y0,
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor=unit_colors[i],
            opacity=0.8,
            xref="x", yref="y", axref="x", ayref="y"
        )

        # Mapped vectors (green)
        x1, y1 = mapped_vectors[:, i]
        fig.add_scatter(x=np.array([0, x1]),
                        y=np.array([0, y1]),
                        mode='lines',
                        line=dict(color=mapped_colors[i], width=1),
                        showlegend=False)
        fig.add_annotation(
            x=x1, y=y1,
            ax=x0, ay=y0,
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor=mapped_colors[i],
            opacity=0.8,
            xref="x", yref="y", axref="x", ayref="y"
        )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    st.markdown(r"""The unit circle vectors are transformed by matrix $A$ to form
        the red ellipsis. Note: the sense of rotation (order of the vectors) is 
        changed from counter-clockwise to clockwise during the transformation; the
        matrix' determinant has negative sign.""")


with col2:

    fig2 = go.Figure()

    # --- Layout ---
    fig2.update_layout(
        width=700,
        height=700,
        xaxis=dict(scaleanchor="y", scaleratio=1, range=[-2, 2], zeroline=True),
        yaxis=dict(range=[-2, 2], zeroline=True),
        template="plotly_white",
        title="Mapping via V<sup>T</sup>",
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
    )


    fig2.add_trace(go.Scatter(
        x=Vtx[0],
        y=Vtx[1],
        fill="toself",
        mode="lines",
        line=dict(color=V_colors[50], width=1),
        fillcolor=V_colors[50],
        opacity=0.5,
        name="Unit circle mapped via V<sup>T</sup>"
    ))

    # --- Draw arrows (vectors) ---
    arrow_step = max(1, initial_vectors.shape[1] // 30)  # ~30 arrows for clarity

    for i in range(0, initial_vectors.shape[1], arrow_step):
        # Unit circle vectors (purple)
        x0, y0 = 0, 0

        # Mapped vectors (green)
        x1, y1 = Vtx[:, i]
        fig2.add_scatter(x=np.array([0, x1]),
                        y=np.array([0, y1]),
                        mode='lines',
                        line=dict(color=V_colors[i], width=1),
                        showlegend=False)
        fig2.add_annotation(
            x=x1, y=y1,
            ax=x0, ay=y0,
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor=V_colors[i],
            opacity=0.8,
            xref="x", yref="y", axref="x", ayref="y"
        )

    st.plotly_chart(fig2, use_container_width=True, key="svd_subplots")
    st.markdown("---")
    st.markdown(r"""The unit circle vectors are mapped by the first part of the SVD:
        the matrix $V^T$. This matrix does not change the shape of the image (it's still
        a circle), but the unit circle vectors are **rotated** under the transformation.""")


with col3:

    fig3 = go.Figure()

    # --- Layout ---
    fig3.update_layout(
        width=700,
        height=700,
        xaxis=dict(scaleanchor="y", scaleratio=1, range=[-2, 2], zeroline=True),
        yaxis=dict(range=[-2, 2], zeroline=True),
        template="plotly_white",
        title="Mapping via ΣV<sup>T</sup>",
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
    )


    fig3.add_trace(go.Scatter(
        x=SigmaVtx[0],
        y=SigmaVtx[1],
        fill="toself",
        mode="lines",
        line=dict(color=Sigma_colors[50], width=1),
        fillcolor=Sigma_colors[50],
        opacity=0.5,
        name="Unit circle mapped via ΣV<sup>T</sup>"
    ))

    # --- Draw arrows (vectors) ---
    arrow_step = max(1, initial_vectors.shape[1] // 30)  # ~30 arrows for clarity

    for i in range(0, initial_vectors.shape[1], arrow_step):
        # Unit circle vectors (purple)
        x0, y0 = 0, 0

        # Mapped vectors (green)
        x1, y1 = SigmaVtx[:, i]
        fig3.add_scatter(x=np.array([0, x1]),
                        y=np.array([0, y1]),
                        mode='lines',
                        line=dict(color=Sigma_colors[i], width=1),
                        showlegend=False)
        fig3.add_annotation(
            x=x1, y=y1,
            ax=x0, ay=y0,
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor=Sigma_colors[i],
            opacity=0.8,
            xref="x", yref="y", axref="x", ayref="y"
        )

    st.plotly_chart(fig3, use_container_width=True, key="svd_subplots2")
    st.markdown("---")
    st.markdown(r"""The unit circle vectors are mapped by the first and second part of
        the SVD: the sequence of $\Sigma V^T$. Here, the image of $V^T$ is stretched 
        along the axes by the singular values (i.e., with the matrix $\Sigma$). In contrast
        to $V^T$, $\Sigma$ does not rotate but rather stretch.""")


with col4:

    fig4 = go.Figure()

    # --- Layout ---
    fig4.update_layout(
        width=700,
        height=700,
        xaxis=dict(scaleanchor="y", scaleratio=1, range=[-2, 2], zeroline=True),
        yaxis=dict(range=[-2, 2], zeroline=True),
        template="plotly_white",
        title="Mapping via UΣV<sup>T</sup>",
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
    )


    fig4.add_trace(go.Scatter(
        x=USigmaVtx[0],
        y=USigmaVtx[1],
        fill="toself",
        mode="lines",
        line=dict(color=U_colors[50], width=1),
        fillcolor=U_colors[50],
        opacity=0.5,
        name="Unit circle mapped via UΣV<sup>T</sup>"
    ))

    # --- Draw arrows (vectors) ---
    arrow_step = max(1, initial_vectors.shape[1] // 30)  # ~30 arrows for clarity

    for i in range(0, initial_vectors.shape[1], arrow_step):
        # Unit circle vectors (purple)
        x0, y0 = 0, 0

        # Mapped vectors (green)
        x1, y1 = USigmaVtx[:, i]
        fig4.add_scatter(x=np.array([0, x1]),
                        y=np.array([0, y1]),
                        mode='lines',
                        line=dict(color=U_colors[i], width=1),
                        showlegend=False)
        fig4.add_annotation(
            x=x1, y=y1,
            ax=x0, ay=y0,
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor=U_colors[i],
            opacity=0.8,
            xref="x", yref="y", axref="x", ayref="y"
        )

    st.plotly_chart(fig4, use_container_width=True, key="svd_subplots3")
    st.markdown("---")
    st.markdown(r"""The unit circle vectors are mapped by the full SVD: the sequence of
        $U\Sigma V^T$. Here, we first have the rotation by $V^T$, then the stretching by
        $\Sigma$, and finally another rotation by $U$. Overall, this sequence then gives
        the same image as the matrix $A$.""")










cont3 = st.container(border=True)
with cont3:
    st.badge("Connection to Eigenvalues and Eigenvectors", color="red")
    st.markdown(r"""
        There is a deep link between **singular values** and **eigenvalues**.
        While eigenvalues are defined only for **square matrices**, singular values 
        extend this concept to **any** rectangular matrix.
        """)

    st.markdown(r"""
        To see this connection, recall that for a square matrix $A$ we solve
        $$A v = \lambda v$$
        to find the eigenpairs $(\lambda, v)$.  
        How could we find the singular values and their left and right singular vectors?  
        We said that $A = U\Sigma V^T$, with $V^T = V^{-1}$ and $U^T = U^{-1}$ because 
        both matrices are othonormal bases of the input resp. output space.  
        Taking a closer look, we see that we might be able to simplify that equation 
        quite a bit by multiplying both sides with $A^T$:
        """)

    st.latex(r"""
        A^T A = (U\Sigma V^T)^T (U\Sigma V^T) = V \Sigma^T U^T U \Sigma V^T = 
            V \Sigma^T \mathbb{I} \Sigma V^T = V \Sigma^T \Sigma V^T = V \Sigma^2 V^T = 
            V \Sigma^2 V^{-1}
        """)
    st.markdown(r"""We know that expression from the eigenvalues and -vectors already!    
        **Specifically**: the matrix $A^TA$ is diagonalizable (can be brought into the form
        $VDV^{-1}$), and the matrix 
        $\Sigma^2$ here corresponds to the eigenvalue matrix $D$ of $A^TA$.  
        $\rightarrow$ $\sigma^2_i$ are the **eigenvalues** of $A^TA$!  
        $spec(A^TA)=\{\lambda_1, \lambda_2\} = \{\sigma_1^2, \sigma_2^2\}$""")

    st.markdown(r"""
        **This means**:
        - the **right singular vectors** $v_i$ of $A$ are **eigenvectors** of $A^T A$  
        - the **left singular vectors** $u_i$ of $A$ are **eigenvectors** of $A A^T$  
        - and the **singular values** $\sigma_i$ of $A$ are the **square roots of the eigenvalues** of $A^T A$.
        """)
    st.latex(r"\sigma_i(A) = \sqrt{\lambda_i(A^TA)}")


cont4 = st.container(border=True)
with cont4:
    st.badge("Metrics and measures based on singular values", color='red')
    st.markdown(r"""As the singular values show how much an input is 'deformed' under
        transformation with a matrix $A$, we can use them to make statements on the 
        mapping.""")
    st.markdown(r"""One measure is the **spectral norm** $||A||_S = \max_i{\sigma_i(A)}$,
        which just corresponds to the largest singular value of the matrix $A$.""")
    st.markdown(r"""Another measure for how 'bad' a matrix deforms unit circle vectors is the 
        **condition of a matrix**:""")
    st.latex(r"""K_S(A)=cond_S(A) = ||A||_S\cdot ||A^{-1}||_S = \frac{\sigma_{max}(A)}{\sigma_{min}(A)}""")
    st.markdown(r"""Here, a condition of $cond_S(A)=1$ would be the **optimum case**: the 
        mapping keeps the circle. **Worst case** would be a condition $cond_S(A)=\infty$, meaning
        the small half-axis (smallest singular value) is $0$ - the transformation 'eats'
        a dimension!""")
    

cont5 = st.container(border=True)
with cont5:
    st.badge("Summary", color="red")
    st.markdown(r"""
        - $U$ and $V$ are **orthogonal**: $U^T U = V^T V = \mathbb{I}$
        - The columns of $U$ are the **left singular vectors** ($u_i$)
        - The columns of $V$ are the **right singular vectors** ($v_i$)
        - The diagonal entries of $\Sigma$ are the **singular values** ($\sigma_i$)
        - Singular values measure how strongly $A$ **stretches** space along certain directions  
        - SVD generalizes the eigenvalue decomposition to **any matrix**, even rectangular ones
        - The singular values can be calculated via $\sigma_i(A) = \sqrt{\lambda_i(A^TA)}$
        """)

    st.markdown(r"""
        Together, they provide a powerful and geometrically intuitive picture of how a matrix 
        acts on space — rotating, stretching, compressing, and possibly collapsing dimensions.
        """)




number_vectors = 100
theta = np.linspace(0, 2*np.pi, number_vectors)
unit_circle = np.array([np.cos(theta), np.sin(theta)])

# Mapping matrix
A = np.array([[0.8, 1],
              [0, -2]])

# Eigenvalues and eigenvectors
eigvals, eigvecs = np.linalg.eig(A)
mapped_eigvecs = eigvecs * eigvals  # scaled by eigenvalues
mapped_vectors = A @ unit_circle

# SVD
U, S, Vt = np.linalg.svd(A)
V = Vt.T
mapped_svd = A @ unit_circle
scaled_left = U * S[np.newaxis, :]  # pre-scaled left singular vectors

# === Plot setup ===
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "Eigenvectors (keep direction, not necessarily orthogonal)",
        "Singular vectors (orthogonal before and after mapping)"
    )
)

# === Left subplot: eigenvectors ===
# Unit circle
fig.add_trace(go.Scatter(
    x=unit_circle[0],
    y=unit_circle[1],
    mode='lines',
    fill='toself',
    fillcolor='rgba(125,125,125,0.2)',
    line=dict(color='rgba(125,125,125,0.8)', width=0.5),
    showlegend=False
), row=1, col=1)

# Mapped circle
fig.add_trace(go.Scatter(
    x=mapped_vectors[0],
    y=mapped_vectors[1],
    mode='lines',
    fill='toself',
    fillcolor='rgba(255,0,0,0.2)',
    line=dict(color='rgba(255,0,0,0.8)', width=0.5),
    showlegend=False
), row=1, col=1)

# Eigenvectors

unit_colors = ['rgb(175,175,175)','rgb(125,125,125)']
mapped_colors = ['rgb(255,0,0)','rgb(180,0,0)']

for k in range(eigvecs.shape[1]):
    fig.add_trace(go.Scatter(
        x=[0, eigvecs[0, k]],
        y=[0, eigvecs[1, k]],
        mode='lines+markers',
        marker=dict(size=8, color=unit_colors[k], symbol='x'),
        line=dict(color=unit_colors[k], width=1.5),
        name=f"eigenvector {k+1}"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=[0, mapped_eigvecs[0, k]],
        y=[0, mapped_eigvecs[1, k]],
        mode='lines+markers',
        marker=dict(size=8, color=mapped_colors[k], symbol='cross'),
        line=dict(color=mapped_colors[k], width=1.5, dash='dash'),
        name=f"mapped eigenvector {k+1}"
    ), row=1, col=1)

# === Right subplot: singular vectors ===
# Right singular vectors (V)
fig.add_trace(go.Scatter(
    x=unit_circle[0],
    y=unit_circle[1],
    mode='lines',
    fill='toself',
    fillcolor='rgba(125,125,125,0.2)',
    line=dict(color='rgba(125,125,125,0.8)', width=0.5),
    showlegend=False
), row=1, col=2)

# Mapped circle
fig.add_trace(go.Scatter(
    x=mapped_vectors[0],
    y=mapped_vectors[1],
    mode='lines',
    fill='toself',
    fillcolor='rgba(255,0,0,0.2)',
    line=dict(color='rgba(255,0,0,0.8)', width=0.5),
    showlegend=False
), row=1, col=2)

unit_colors2 = ['rgb(160,0,200)', 'rgb(100,0,160)']
mapped_colors2 = ['rgb(255,80,180)', 'rgb(200,0,120)']
mapped_colors_norm2 = ['rgb(255,150,100)', 'rgb(255,80,60)']

for k in range(V.shape[1]):
    fig.add_trace(go.Scatter(
        x=[0, V[0, k]],
        y=[0, V[1, k]],
        mode='lines+markers',
        marker=dict(size=8, color=mapped_colors_norm2[k], symbol='x'),
        line=dict(color=mapped_colors_norm2[k], width=1.5),
        name=f"right singular vector {k+1}"
    ), row=1, col=2)


for k in range(U.shape[1]):
    fig.add_trace(go.Scatter(
        x=[0, U[0, k]],
        y=[0, U[1, k]],
        mode='lines+markers',
        marker=dict(size=8, color=unit_colors2[k], symbol='cross'),
        line=dict(color=unit_colors2[k], width=1.5, dash='dash'),
        name=f"left singular vector {k+1}"
    ), row=1, col=2)

    # Scaled left singular vectors (sigma * U)
    fig.add_trace(go.Scatter(
        x=[0, S[k]*U[0, k]],
        y=[0, S[k]*U[1, k]],
        mode='lines+markers',
        marker=dict(size=8, color=mapped_colors2[k], symbol='cross'),
        line=dict(color=mapped_colors2[k], width=1.5, dash='dash'),
        name=f"scaled left singular vector {k+1}"
    ), row=1, col=2)

# === Layout adjustments ===
fig.update_layout(
    showlegend=True,
    height=600,
    width=1000
)

for i in [1,2]:
    fig.update_xaxes(scaleanchor=f"y{i}", dtick=2, tickmode='array', row=1, col=i)
    fig.update_yaxes(scaleanchor=f"x{i}", dtick=2, tickmode='array', row=1, col=i)

cont = st.container(border=True)
with cont:
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(r"""Whereas the eigenvectors **do not** have to be orthogonal to each other
        (they 'only' fulfill similarity before and after mapping),
        the (left and right) singular vectors are orthogonal to each other before **and**
        after mapping (see right figure).  
        **Even more important**: the singular values denote the **length of long and short
        half-axes** of the ellipse that is the image of the unit circle vectors under $A$.  
        The long half-axis corresponds to the **largest singular value**, the short half-axis
        to the **smallest singular value**.""")
    st.markdown(r"""This means: the singular values tell us about maximum and minimum 
        deformation during the mapping with matrix $A$.""")
    
