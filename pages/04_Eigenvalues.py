import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.cm as cm
import matplotlib.colors as mcolors

st.set_page_config(page_title="Eigenvalues Demo", layout="wide")
st.title("Eigenvalues of a Matrix")


A = np.array([[0.8, 1],
              [0, -2]])

st.markdown(r"We consider the matrix $A \in \mathbb{R}^{2 \times 2}$:")
st.latex(r"""
A = 
\begin{pmatrix}
0.8 & 1 \\
0 & -2
\end{pmatrix}
""")

cont = st.container(border=True)

with cont:
    st.badge("The Eigenvalue Equation", color='red')
    st.markdown(r"""
    A vector $v \neq 0$ is called an **eigenvector** of $A$ if there exists a scalar $\lambda$ such that:
    $$
    A v = \lambda v
    $$
    The equation above is called the **Eigenvalue equation**. Here, $\lambda \in \mathbb{C}$ is the corresponding **eigenvalue**, which denotes how much the eigenvector
    is **stretched** or **compressed** during the mapping.""")
    st.markdown("""In words, the **Eigenvalue Equation** states that we look for a vector $v$
        that **keeps its direction** even after being mapped with matrix $A$ and at most changes in length.
    """)
    st.markdown(r"""To ensure that we don't end up with the trivial solution ($v=0$), we actively exclude it 
        from the possible solutions, as seen in the first line. """)


cont = st.container(border=True)
with cont:
    st.badge("Solving the Eigenvalue Equation", color='red')
    st.markdown(r"""To have everything more compact, we can bring the right side of the
        equation over to the left side:""")
    st.latex(r"(A - \lambda I)v = 0")
    st.markdown(r"""where $I$ denotes the identity matrix.""")
    st.markdown(r"""This should remind you of our discussions of the **kernel** 
        of a matrix, $ker(A)$ - with this equations, we are looking **exactly** for vectors
        that map to the zero vector. As we want the vector $v\neq 0$, we **further** need
        the kernel of A to be **non-trivial**, i.e., it should hold vectors that are **not only**
        the zero vector! From our previous discussion we now know that for this to hold, 
        $\tilde{A} = (A-\lambda I)$ needs to be **singular**.""")
    st.markdown(r"""How do we enforce that the matrix $\tilde{A}$ is singular, now?""")
    st.markdown(r"""$\rightarrow$ Via its determinant. We simply require that: """)
    st.latex(r"""
        det(A - \lambda I) = 
            \left|\begin{pmatrix}
            a_{11} & a_{12} \\
            a_{21} & a_{22}
            \end{pmatrix} - 
            \begin{pmatrix}
            \lambda & 0 \\
            0 & \lambda
            \end{pmatrix}\right| = \left|\begin{pmatrix}
            a_{11}-\lambda & a_{12} \\
            a_{21} & a_{22} - \lambda
            \end{pmatrix}\right| = 
             (a_{11}-\lambda)\cdot (a_{22}-\lambda) - \left( 
            a_{21}\cdot a_{12}  
            \right) = 
        0""")


cont = st.container(border=True)
with cont:
    st.badge("The Characteristic Polynomial & Finding the Eigenvalues", color='red')
    st.markdown(r"""The last equation above is called the **characteristic equation**:""")
    st.latex(r"""
    \det(\tilde{A}) = \det(A - \lambda I) = (a_{11}-\lambda)(a_{22}-\lambda) - a_{21} a_{12} =
    \underbrace{\lambda^2 - (a_{11}+a_{22})\lambda + (a_{11}a_{22}-a_{12}a_{21})}_{\text{characteristic polynomial}}
    = 0
    """)
    st.markdown(r"""and the last part of the equation is called the **characteristic 
        polynomial** of $\tilde{A}.$""")
    st.markdown(r"""The solutions $\lambda_1, \lambda_2$ of this quadratic equation
        (for $\tilde{A} \in \mathbb{R}^{2 \times 2}$) are exactly the **roots of the 
        characteristic polynomial**.""")
    st.markdown(r"""For our matrix from above, the characteristic polynomial looks like:""")
    st.latex(r"""
        det(A - \lambda I) = \left|
        \begin{pmatrix}
        0.8 - \lambda & 1 \\
        0 & -2 - \lambda
        \end{pmatrix} \right| = 
        (0.8 - \lambda)(-2 - \lambda) = \lambda^2 + 1.2 \lambda - 1.6 = 0, 
    """)
    st.markdown(r"""and the corresponding solutions (the **eigenvalues**) are:""")
    eigvals = np.linalg.eigvals(A)
    st.latex(rf"""
    \lambda_1 = {eigvals[0]}, \quad
    \lambda_2 = {eigvals[1]}
    """)




cont = st.container(border=True)
with cont:
    st.badge("Finding the Eigenvectors", color='red')
    st.markdown(r"""To find the corresponding eigenvectors, we now insert the eigenvalues
        into the Eigenvalue Equation one by one.""")
    st.markdown(r"For $\lambda_1 = 0.8$:")
    st.latex(r"""
        (A - \lambda_1 I)v = \begin{pmatrix}
            0.8 - \lambda_1 & 1 \\
            0 & - 2 - \lambda_1
            \end{pmatrix} 
            \begin{pmatrix}
            v_1 \\
            v_2
            \end{pmatrix} = 
            \begin{pmatrix}
            0 \cdot v_1 + 1 \cdot v_2 \\
            0 \cdot v_1 - 2.8 \cdot v_2
            \end{pmatrix} =
            \begin{pmatrix}
            0 \\
            0
            \end{pmatrix}
    """)
    st.markdown(r"""From the first row: $1\cdot v_2 = 0 \Rightarrow v_2 = 0$. We can
        chose $v_1$ however we like (no restrictions).""")
    st.markdown(r"""A possible eigenvector: $\vec{v}_1 = \begin{pmatrix}
        1 \\
        0
        \end{pmatrix}
    $""")
    st.markdown(r"For $\lambda_2 = -2$:")
    st.latex(r"""
        (A - \lambda_2 I)v = \begin{pmatrix}
            0.8 - \lambda_2 & 1 \\
            0 & - 2 - \lambda_2
            \end{pmatrix} 
            \begin{pmatrix}
            v_1 \\
            v_2
            \end{pmatrix} = 
            \begin{pmatrix}
            2.8 \cdot v_1 + 1 \cdot v_2 \\
            0 \cdot v_1 + 0 \cdot v_2
            \end{pmatrix} =
            \begin{pmatrix}
            0 \\
            0
            \end{pmatrix}
    """)
    st.markdown(r"""From the first row: $2.8\cdot v_1 + 1\cdot v_2 = 0 \Rightarrow v_2 = -2.8v_1$.""")
    st.markdown(r"""A possible eigenvector (normed): $\vec{v}_2 = \frac{1}{\sqrt{1^2 + 2.8^2}}\begin{pmatrix}
        1 \\
        -2.8
        \end{pmatrix}
    $""")

theta = np.linspace(0, 2*np.pi, 100)
unit_vectors = np.vstack((np.cos(theta), np.sin(theta)))
mapped_vectors = A @ unit_vectors

eigenvalues, eigenvectors = np.linalg.eig(A)
mapped_eigenvectors = eigenvalues * eigenvectors

cont = st.container(border=True)
with cont:
    st.badge("Spectrum and spectral radius of a matrix", color='red')
    st.markdown(r"""The set of all eigenvalues of a matrix $A$ is called the **spectrum**
        or **spectral set** $spec(A)$ of $A$.   
        In the case of our matrix, the spectrum corresponds to:   
        $spec(A) = \{ \lambda_1, \lambda_2 \} = \{0.8, -2\}$""")
    st.markdown(r"""The **largest eigenvalue** in terms of **absolute value** of a matrix
        $A$ is called the **spectral radius** $\rho(A)$ of $A$.  
        In the case of our matrix, the spectral radius is:   
        $\rho(A) = \max_i(|\lambda_i|) = 2$""")


cont = st.container(border=True)
with cont:
    st.badge("Recap and important facts about eigenvalues and eigenvectors", color='red')
    st.markdown(r"""
    **Basic facts**
    - Eigenvectors point in directions that are **invariant** under the transformation by $A$ — only their **length (and possibly orientation)** changes, not their direction.
    - Eigenvalues $\lambda$ describe how much these directions are **stretched**, **compressed**, or **flipped**.

    ---

    **Basic properties**
    - The **trace** of $A$ (sum of diagonal elements) equals the **sum** of the eigenvalues:
    $\mathrm{tr}(A) = \sum_i \lambda_i$.
    - The **determinant** of $A$ equals the **product** of the eigenvalues:
    $\det(A) = \prod_i \lambda_i$.
    - A matrix is **invertible (regular)** if and only if none of its eigenvalues are zero
    - The eigenvalues of $A^{-1}$ are $\dfrac{1}{\lambda_i}$ (this works because, if $A$
        is invertible (regular), none of its eigenvalues are zero)

    ---

    **Some facts that make our lives a lot easier in practice**
    - **Triangular (or diagonal) matrices**:
        The eigenvalues are simply the entries on the **main diagonal**.  
        E.g., for  
            $A = 
                \begin{pmatrix} 
                2 & 1 \\
                0 & 3
                \end{pmatrix}
            $, $spec(A) = \{2,3\}$, or for  
            $A = 
                \begin{pmatrix} 
                3 & 0 \\
                0 & 5
                \end{pmatrix}
            $, $spec(A) = \{3,5\}$  
        $\rightarrow$ we don't need to calculate for ages, we can just read it off the diagonal
                
    - **Symmetric matrices** ($A = A^\top$) always have **real eigenvalues**.  
    - for real matrices, **complex eigenvalues** always appear in **conjugate pairs**:  
        if $\lambda = a + ib \in \mathbb{C}$ is an eigenvalue, then $\bar{\lambda} = a -ib$ is an eigenvalue 

    ---

    **Interpretation and geometry**
    - if $|\lambda| > 1$ → vectors along $v$ are **stretched**.
    - if $|\lambda| < 1$ → vectors are **contracted**.
    - if $\lambda < 0$ → vectors **flip orientation**.
    - if $\lambda = 0$ → the direction **collapses to the origin** (dimension loss).
    - if all $\lambda = 0$ → the transformation sends **every vector to $\mathbf{0}$**.

    ---

    **The spectrum and spectral radius**
    - the spectral radius determines **asymptotic behavior**:
    - if $\rho(A) < 1$, then $A^k \to 0$ as $k \to \infty$.  
    - if $\rho(A) > 1$, then powers of $A$ **blow up**.

    ---

    **Some more extra insights**
    - if $A$ has $n$ linearly independent eigenvectors → $A$ is **diagonalizable** because  
        $AV = V\Lambda \Leftrightarrow A = V \Lambda V^{-1}$
    - **Defective** matrices lack enough eigenvectors for diagonalization (common in repeated eigenvalues)
    - For stochastic matrices (Markov chains):
    $\rho(A) = 1$ always.
    ---
    """)




cont = st.container(border=True)
with cont:
    st.badge("Interpreting matrix mapping figures", color='red')
    fig = go.Figure()

    fig.update_layout(
        xaxis=dict(
            scaleanchor="y",  # Ensure equal scaling for x and y axes
            dtick=2,  # Set grid tick spacing (integer steps)
            tickmode='array',  # Use specific tick values
            #tickvals=np.arange(-10,10)
        ),
        yaxis=dict(
            scaleanchor="x",  # Ensure equal scaling for x and y axes
            dtick=2,  # Set grid tick spacing (integer steps)
            tickmode='array',  # Use specific tick values,  # Explicitly set tick positions
            #tickvals=np.arange(-10,10)
        ),
    )

    fig.add_trace(go.Scatter(
        x=unit_vectors[0],
        y=unit_vectors[1],
        fill="toself",
        mode="lines",
        line=dict(color='rgba(125,125,125,0.5)', width=0.5),
        fillcolor='rgba(125,125,125,0.2)',
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=mapped_vectors[0],
        y=mapped_vectors[1],
        fill="toself",
        mode="lines",
        line=dict(color='rgb(255,0,0)', width=0.5),
        fillcolor='rgba(255,0,0,0.2)',
        showlegend=False
    ))

    unit_colors = ['rgb(175,175,175)','rgb(125,125,125)']
    mapped_colors = ['rgb(255,0,0)','rgb(180,0,0)']

    for k in range(0,len(eigenvectors)):
        x, y = eigenvectors[:, k]
        xm, ym = mapped_eigenvectors[:, k]
        fig.add_scatter(x=np.array([0, eigenvectors[0, k]]),
                        y=np.array([0, eigenvectors[1, k]]),
                        mode='lines',
                        line=dict(color=unit_colors[k], width=1.5),
                        showlegend=False
                        )
        fig.add_scatter(x=np.array([0, mapped_eigenvectors[0, k]]),
                        y=np.array([0, mapped_eigenvectors[1, k]]),
                        mode='lines',
                        line=dict(color=mapped_colors[k], width=1.5),
                        showlegend=False
                        )
        fig.add_annotation(
            x=x, y=y,
            ax=0, ay=0,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=1,           # smaller arrowhead
            arrowsize=0.75,           # smaller size
            arrowwidth=1.5,
            arrowcolor=unit_colors[k],
        )
        fig.add_annotation(
            x=xm, y=ym,
            ax=0, ay=0,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=1,           # smaller arrowhead
            arrowsize=0.75,           # smaller size
            arrowwidth=1.5,
            arrowcolor=mapped_colors[k],
        )
        

    fig.update_layout(title="Mapping of the eigenvectors")

    col1, col2 = st.columns([0.6,0.4])

    with col1:
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(r"""
        **Interpreting a figure like this**:

        - we see two eigenvectors $\rightarrow A \in \mathbb{R}^{2 \times 2}$
        - the image is a full ellipse (no collapsing to a line or dot) $\rightarrow$
        none of the eigenvalues is $0$ $\Leftrightarrow$ matrix $A$ is regular
        - one eigenvector points into a different **room** direction (attention, **not**
        mathematical direction - it's still on the same line) after mapping $\rightarrow$ that eigenvalue is **negative**
        - one eigenvector is $\approx$ two times as long after mapping $\rightarrow$ the absolute
            value of the corresponding eigenvalue is $\approx 2$.
        - one eigenvector is a little shorter after mapping $\rightarrow$ the absolute value
            of the corresp. eigenvalue is $\approx 0.8$
        """)



    B = np.array([[0.8, 1],
                [0.4, 0.5]])


    mapped_vectors = B @ unit_vectors

    eigenvalues, eigenvectors = np.linalg.eig(B)
    mapped_eigenvectors = eigenvalues * eigenvectors


    fig = go.Figure()

    fig.update_layout(
        xaxis=dict(
            scaleanchor="y",  # Ensure equal scaling for x and y axes
            dtick=2,  # Set grid tick spacing (integer steps)
            tickmode='array',  # Use specific tick values
            #tickvals=np.arange(-10,10)
        ),
        yaxis=dict(
            scaleanchor="x",  # Ensure equal scaling for x and y axes
            dtick=2,  # Set grid tick spacing (integer steps)
            tickmode='array',  # Use specific tick values,  # Explicitly set tick positions
            #tickvals=np.arange(-10,10)
        ),
    )

    fig.add_trace(go.Scatter(
        x=unit_vectors[0],
        y=unit_vectors[1],
        fill="toself",
        mode="lines",
        line=dict(color='rgba(125,125,125,0.5)', width=0.5),
        fillcolor='rgba(125,125,125,0.2)',
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=mapped_vectors[0],
        y=mapped_vectors[1],
        fill="toself",
        mode="lines",
        line=dict(color='rgb(255,0,0)', width=0.5),
        fillcolor='rgba(255,0,0,0.2)',
        showlegend=False
    ))

    unit_colors = ['rgb(175,175,175)','rgb(125,125,125)']
    mapped_colors = ['rgb(255,0,0)','rgb(180,0,0)']

    for k in range(0,len(eigenvectors)):
        x, y = eigenvectors[:, k]
        xm, ym = mapped_eigenvectors[:, k]
        fig.add_scatter(x=np.array([0, eigenvectors[0, k]]),
                        y=np.array([0, eigenvectors[1, k]]),
                        mode='lines',
                        line=dict(color=unit_colors[k], width=1.5),
                        showlegend=False
                        )
        fig.add_scatter(x=np.array([0, mapped_eigenvectors[0, k]]),
                        y=np.array([0, mapped_eigenvectors[1, k]]),
                        mode='lines',
                        line=dict(color=mapped_colors[k], width=1.5),
                        showlegend=False
                        )
        fig.add_annotation(
            x=x, y=y,
            ax=0, ay=0,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=1,           # smaller arrowhead
            arrowsize=0.75,           # smaller size
            arrowwidth=1.5,
            arrowcolor=unit_colors[k],
        )
        fig.add_annotation(
            x=xm, y=ym,
            ax=0, ay=0,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=1,           # smaller arrowhead
            arrowsize=0.75,           # smaller size
            arrowwidth=1.5,
            arrowcolor=mapped_colors[k],
        )
        

    fig.update_layout(title="Mapping of the eigenvectors")

    col1, col2 = st.columns([0.6,0.4])

    with col1:
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(r"""
        **Interpreting a figure like this**:

        - we see two eigenvectors $\rightarrow A \in \mathbb{R}^{2 \times 2}$
        - the image is **not** a full ellipse, the image of the unit circle under this matrix
            collapses to a **line** $\rightarrow$
            one of the eigenvalues is $0$ $\Leftrightarrow$ matrix $A$ is singular
            $\Leftrightarrow$ matrix $A$ is **not** invertible!
        - one eigenvector is mapped to zero after mapping $\rightarrow$ the corresponding
            eigenvalue is $0$
        - one eigenvector gets $\approx$ 30\% longer after mapping $\rightarrow$ the absolute
            value of the corresponding eigenvalue is $\approx$ 1.3.
        """)



    C = np.array([[0, 1],
                [0, 0]])


    mapped_vectors = C @ unit_vectors

    eigenvalues, eigenvectors = np.linalg.eig(C)
    mapped_eigenvectors = eigenvalues * eigenvectors


    fig = go.Figure()

    fig.update_layout(
        xaxis=dict(
            scaleanchor="y",  # Ensure equal scaling for x and y axes
            dtick=2,  # Set grid tick spacing (integer steps)
            tickmode='array',  # Use specific tick values
            #tickvals=np.arange(-10,10)
        ),
        yaxis=dict(
            scaleanchor="x",  # Ensure equal scaling for x and y axes
            dtick=2,  # Set grid tick spacing (integer steps)
            tickmode='array',  # Use specific tick values,  # Explicitly set tick positions
            #tickvals=np.arange(-10,10)
        ),
    )

    fig.add_trace(go.Scatter(
        x=unit_vectors[0],
        y=unit_vectors[1],
        fill="toself",
        mode="lines",
        line=dict(color='rgba(125,125,125,0.5)', width=0.5),
        fillcolor='rgba(125,125,125,0.2)',
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=mapped_vectors[0],
        y=mapped_vectors[1],
        fill="toself",
        mode="lines",
        line=dict(color='rgb(255,0,0)', width=0.5),
        fillcolor='rgba(255,0,0,0.2)',
        showlegend=False
    ))

    unit_colors = ['rgb(175,175,175)','rgb(125,125,125)']
    mapped_colors = ['rgb(255,0,0)','rgb(180,0,0)']

    for k in range(0,len(eigenvectors)):
        x, y = eigenvectors[:, k]
        xm, ym = mapped_eigenvectors[:, k]
        fig.add_scatter(x=np.array([0, eigenvectors[0, k]]),
                        y=np.array([0, eigenvectors[1, k]]),
                        mode='lines',
                        line=dict(color=unit_colors[k], width=1.5),
                        showlegend=False
                        )
        fig.add_scatter(x=np.array([0, mapped_eigenvectors[0, k]]),
                        y=np.array([0, mapped_eigenvectors[1, k]]),
                        mode='lines',
                        line=dict(color=mapped_colors[k], width=1.5),
                        showlegend=False
                        )
        fig.add_annotation(
            x=x, y=y,
            ax=0, ay=0,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=1,           # smaller arrowhead
            arrowsize=0.75,           # smaller size
            arrowwidth=1.5,
            arrowcolor=unit_colors[k],
        )
        fig.add_annotation(
            x=xm, y=ym,
            ax=0, ay=0,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=1,           # smaller arrowhead
            arrowsize=0.75,           # smaller size
            arrowwidth=1.5,
            arrowcolor=mapped_colors[k],
        )
        

    fig.update_layout(title="Mapping of the eigenvectors")

    col1, col2 = st.columns([0.6,0.4])

    with col1:
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(r"""
        **Interpreting a figure like this**:

        - we see two eigenvectors $\rightarrow A \in \mathbb{R}^{2 \times 2}$
        - the image is **not** a full ellipse and **not** even a line - the image of the unit circle 
            under this matrix collapses into a **dot** $\rightarrow$
            both of the eigenvalues are $0$ $\Leftrightarrow$ matrix $A$ is singular
            $\Leftrightarrow$ matrix $A$ is **not** invertible!
        - both eigenvectors are mapped to zero after mapping $\rightarrow$ the corresponding
            eigenvalues are $0$
        """)