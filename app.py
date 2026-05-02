"""
app.py  —  Interface Streamlit FER2013
Lancer avec : streamlit run app.py
"""

import os
import numpy as np
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

from predictor import predict_all, MODEL_STATS, EMOTIONS
from models import EMOTIONS as EMOTION_LIST

# ── Configuration de la page ──────────────────────────────────────────────────
st.set_page_config(
    page_title="FER2013 — Reconnaissance d'Émotions",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personnalisé ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1A3A5C;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f0f4f8;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .emotion-badge {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1rem;
    }
    .winner-card {
        border: 2px solid #2E75B6;
        border-radius: 12px;
        padding: 1rem;
        background: #EBF3FB;
    }
    .unavailable {
        color: #999;
        font-style: italic;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Émojis par émotion ────────────────────────────────────────────────────────
EMOTION_EMOJI = {
    "angry":    "😠",
    "disgust":  "🤢",
    "fear":     "😨",
    "happy":    "😄",
    "sad":      "😢",
    "surprise": "😲",
    "neutral":  "😐",
}

EMOTION_COLOR = {
    "angry":    "#E24B4A",
    "disgust":  "#639922",
    "fear":     "#7F77DD",
    "happy":    "#EF9F27",
    "sad":      "#378ADD",
    "surprise": "#D4537E",
    "neutral":  "#888780",
}

MODEL_COLORS = {
    "Custom CNN":       "#378ADD",
    "VGG-Light":        "#1D9E75",
    "ResNet18-Adapted": "#D85A30",
    "Mini-Xception":    "#7F77DD",
}

RADAR_MODEL_COLORS = {
    "Custom CNN":       "#184F92",
    "VGG-Light":        "#0F5D45",
    "ResNet18-Adapted": "#8F3118",
    "Mini-Xception":    "#4E45A0",
}

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    models_dir = st.text_input(
        "Dossier des modèles (.pth)",
        value="saved_models",
        help="Chemin vers le dossier contenant les fichiers model_*.pth"
    )

    st.markdown("---")
    st.markdown("### Modèles disponibles")

    # Vérification rapide de la présence des fichiers
    from predictor import MODEL_REGISTRY
    for name, info in MODEL_REGISTRY.items():
        path = os.path.join(models_dir, info["file"])
        if os.path.exists(path):
            st.markdown(f"✅ **{name}**")
        else:
            st.markdown(f"❌ {name} `({info['file']})`")

    confidence_threshold = 30
    top_k = 5

    st.markdown("---")
    st.markdown("### 📊 Statistiques d'entraînement")
    for name, stats in MODEL_STATS.items():
        st.markdown(f"**{name}** — {stats['val_acc']:.1f}% val acc")

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🧠 Reconnaissance des Émotions Faciales</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">FER2013 · 7 émotions · Comparaison de 4 architectures CNN · PyTorch</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
col_upload, col_info = st.columns([1, 2])

with col_upload:
    uploaded = st.file_uploader(
        "Chargez une image faciale",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="L'image sera convertie en niveaux de gris 64×64 avec CLAHE."
    )

with col_info:
    if uploaded is None:
        st.info("**Comment utiliser cette interface :**\n\n"
                "1. Chargez une image faciale via le bouton à gauche.\n"
                "2. Tous les modèles disponibles effectueront leur prédiction.\n"
                "3. Comparez les résultats dans les onglets ci-dessous.")

# ─────────────────────────────────────────────────────────────────────────────
# PRÉDICTION
# ─────────────────────────────────────────────────────────────────────────────
if uploaded is not None:
    pil_image = Image.open(uploaded).convert("RGB")

    with st.spinner("Inférence en cours sur les 4 modèles…"):
        results = predict_all(pil_image, models_dir=models_dir)

    tensor    = results.pop("_tensor")
    available = {k: v for k, v in results.items() if v.get("available")}
    missing   = {k: v for k, v in results.items() if not v.get("available")}
    best_model_name = "VGG-Light"
    best_result = available.get(best_model_name)

    # ── Aperçu image + prédiction du meilleur modèle ────────────────────────
    st.markdown("---")
    col_img, col_pred = st.columns([1, 3])

    with col_img:
        st.image(pil_image, caption="Image originale", use_container_width=True)
        # Image prétraitée
        import cv2
        import numpy as np
        pil_gray = pil_image.convert("L")
        arr = np.array(pil_gray)
        arr = cv2.resize(arr, (64, 64), interpolation=cv2.INTER_AREA)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        arr_clahe = clahe.apply(arr)
        st.image(arr_clahe, caption="Prétraitée (64×64 CLAHE)", use_container_width=True)

    with col_pred:
        if best_result is None:
            st.error("Aucun modèle disponible. Vérifiez le dossier des modèles.")
        else:
            best_val_acc = MODEL_STATS.get(best_model_name, {}).get("val_acc", 0)

            st.markdown(f"""
            <div class="winner-card">
                <div style="font-size:0.85rem; color:#555; margin-bottom:0.4rem;">
                    Résultat final de finetuned_vgg_light.pth
                </div>
                <div style="font-size:0.9rem; color:#555; margin-bottom:0.2rem;">
                    {best_model_name} · {best_val_acc:.2f}% val acc
                </div>
                <div style="font-size:2.5rem; margin-bottom:0.2rem;">
                    {EMOTION_EMOJI.get(best_result["predicted"], "❓")}
                </div>
                <div style="font-size:1.6rem; font-weight:700; color:#1A3A5C;">
                    {best_result["predicted"].upper()}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Résumé rapide par modèle
            st.markdown("#### Prédictions par modèle")
            cols_pred = st.columns(len(available))
            for col, (name, res) in zip(cols_pred, available.items()):
                conf = res["confidence"] * 100
                em   = res["predicted"]
                color = "#2E75B6" if conf >= confidence_threshold else "#E24B4A"
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:0.78rem; color:#555; margin-bottom:4px;">{name}</div>
                        <div style="font-size:1.4rem;">{EMOTION_EMOJI.get(em,'')}</div>
                        <div style="font-weight:600; color:{color};">{em}</div>
                        <div style="font-size:0.8rem; color:#666;">{conf:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # ONGLETS DÉTAILLÉS
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    tab_names = ["📊 Comparaison globale"] + [f"🔍 {n}" for n in available] + ["ℹ️ Modèles"]
    tabs = st.tabs(tab_names)

    # ── TAB 0 : Comparaison globale ───────────────────────────────────────────
    with tabs[0]:
        st.markdown("### Distribution des probabilités — tous les modèles")

        # Grouped bar chart
        fig_group = go.Figure()

        for name, res in available.items():
            probs_pct = [round(res["probs"][EMOTION_LIST.index(e)] * 100, 1) for e in EMOTION_LIST]
            fig_group.add_trace(go.Bar(
                name=name,
                x=EMOTION_LIST,
                y=probs_pct,
                marker_color=MODEL_COLORS[name],
                text=[f"{p:.1f}%" for p in probs_pct],
                textposition="outside",
                textfont=dict(size=10),
            ))

        fig_group.update_layout(
            barmode="group",
            xaxis_title="Émotion",
            yaxis_title="Probabilité (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=420,
            margin=dict(t=60, b=40, l=40, r=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig_group.add_hline(y=confidence_threshold, line_dash="dot",
                            line_color="gray", annotation_text=f"Seuil {confidence_threshold}%")
        st.plotly_chart(fig_group, use_container_width=True)

        # Radar chart
        st.markdown("### Radar — profil émotionnel par modèle")
        fig_radar = go.Figure()
        theta = EMOTION_LIST + [EMOTION_LIST[0]]
        for name, res in available.items():
            r = [round(res["probs"][EMOTION_LIST.index(e)] * 100, 1) for e in EMOTION_LIST]
            r.append(r[0])
            fig_radar.add_trace(go.Scatterpolar(
                r=r, theta=theta, name=name,
                line=dict(color=RADAR_MODEL_COLORS[name], width=3),
                fill="toself", opacity=0.3
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
            margin=dict(t=60, b=40, l=40, r=40),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Heatmap
        st.markdown("### Heatmap — probabilités (modèles × émotions)")
        heatmap_data = np.array([
            [round(res["probs"][EMOTION_LIST.index(e)] * 100, 1) for e in EMOTION_LIST]
            for res in available.values()
        ])
        fig_heat = go.Figure(go.Heatmap(
            z=heatmap_data,
            x=EMOTION_LIST,
            y=list(available.keys()),
            colorscale="Blues",
            text=[[f"{v:.1f}%" for v in row] for row in heatmap_data],
            texttemplate="%{text}",
            textfont=dict(size=11),
            colorbar=dict(title="Prob (%)"),
        ))
        fig_heat.update_layout(
            height=260,
            margin=dict(t=20, b=40, l=140, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # ── TABS individuels par modèle ────────────────────────────────────────────
    for tab_idx, (name, res) in enumerate(available.items(), start=1):
        with tabs[tab_idx]:
            probs = res["probs"]
            pred  = res["predicted"]
            conf  = res["confidence"] * 100

            col_a, col_b = st.columns([1, 2])

            with col_a:
                # Carte résultat
                conf_color = "#2E75B6" if conf >= confidence_threshold else "#E24B4A"
                st.markdown(f"""
                <div style="border:2px solid {conf_color}; border-radius:12px; padding:1.2rem; text-align:center; margin-bottom:1rem;">
                    <div style="font-size:3rem;">{EMOTION_EMOJI.get(pred,'❓')}</div>
                    <div style="font-size:1.4rem; font-weight:700; color:#1A3A5C;">{pred.upper()}</div>
                    <div style="font-size:1rem; color:{conf_color}; font-weight:600;">{conf:.1f}% de confiance</div>
                    {"⚠️ Confiance faible" if conf < confidence_threshold else "✅ Confiance suffisante"}
                </div>
                """, unsafe_allow_html=True)

                # Méta modèle
                stats = MODEL_STATS[name]
                st.markdown(f"""
                <div class="metric-card">
                    <b>Paramètres</b>: {stats['params']:,}<br>
                    <b>Val accuracy</b>: {stats['val_acc']}%<br>
                    <b>Époques</b>: {stats['epochs']}
                </div>
                """, unsafe_allow_html=True)

            with col_b:
                # Top-K barres horizontales
                idx_sorted = np.argsort(probs)[::-1][:top_k]
                emotions_sorted = [EMOTION_LIST[i] for i in idx_sorted]
                probs_sorted    = [round(probs[i] * 100, 1) for i in idx_sorted]
                colors_sorted   = [EMOTION_COLOR.get(e, "#888") for e in emotions_sorted]

                fig_bar = go.Figure(go.Bar(
                    x=probs_sorted,
                    y=[f"{EMOTION_EMOJI.get(e,'')} {e}" for e in emotions_sorted],
                    orientation="h",
                    marker_color=colors_sorted,
                    text=[f"{p:.1f}%" for p in probs_sorted],
                    textposition="outside",
                ))
                fig_bar.add_vline(x=confidence_threshold, line_dash="dot", line_color="gray")
                fig_bar.update_layout(
                    title=f"Top-{top_k} prédictions — {name}",
                    xaxis_title="Probabilité (%)",
                    xaxis_range=[0, max(probs_sorted) * 1.25],
                    height=320,
                    margin=dict(t=50, b=30, l=120, r=60),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

    # ── TAB : Info modèles ────────────────────────────────────────────────────
    with tabs[-1]:
        st.markdown("### Architectures comparées")

        data_table = {
            "Modèle":           list(MODEL_STATS.keys()),
            "Paramètres":       [f"{v['params']:,}" for v in MODEL_STATS.values()],
            "Val Accuracy (%)": [f"{v['val_acc']:.2f}" for v in MODEL_STATS.values()],
            "Époques":          [str(v['epochs']) for v in MODEL_STATS.values()],
            "Fichier":          [MODEL_REGISTRY[k]["file"] for k in MODEL_STATS],
            "Disponible":       ["✅" if os.path.exists(os.path.join(models_dir, MODEL_REGISTRY[k]["file"])) else "❌"
                                 for k in MODEL_STATS],
        }
        import pandas as pd
        st.dataframe(pd.DataFrame(data_table), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### Résultats d'entraînement (référence notebook)")

        fig_acc = go.Figure(go.Bar(
            x=list(MODEL_STATS.keys()),
            y=[v["val_acc"] for v in MODEL_STATS.values()],
            marker_color=[MODEL_COLORS[k] for k in MODEL_STATS],
            text=[f"{v['val_acc']:.1f}%" for v in MODEL_STATS.values()],
            textposition="outside",
        ))
        fig_acc.update_layout(
            yaxis_title="Validation Accuracy (%)",
            yaxis_range=[0, 80],
            height=360,
            margin=dict(t=40, b=40, l=50, r=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_acc, use_container_width=True)

        if missing:
            st.markdown("---")
            st.warning(f"**Modèles introuvables** dans `{models_dir}` :")
            for name in missing:
                st.markdown(f"- `{MODEL_REGISTRY[name]['file']}`")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#aaa; font-size:0.8rem;'>"
    "FER2013 · 4ème année Génie Logiciel · Traitement d'Images"
    "</div>",
    unsafe_allow_html=True
)
