
import math
import re
from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Writing DNA", page_icon="🧬", layout="wide")

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
STOPWORDS = set(
    """a about above after again all also am an and any are as at be because been
    before being below between both but by can could did do does doing down during
    each few for from further had has have having he her here hers him his how i if
    in into is it its just me more most my no nor not now of off on once only or
    other our out over own same she should so some such than that the their them
    then there these they this those through to too under until up very was we were
    what when where which while who whom why will with would you your yours""".split()
)

TRANSITIONS = {
    "however", "therefore", "moreover", "furthermore", "meanwhile", "thus", "hence",
    "consequently", "nevertheless", "nonetheless", "instead", "otherwise", "also",
    "besides", "similarly", "likewise", "finally", "first", "second", "third",
    "next", "then", "still", "yet", "although", "though", "whereas", "because",
    "since", "unless", "while", "but", "and", "so", "or",
}
HEDGES = {
    "maybe", "perhaps", "possibly", "probably", "somewhat", "sort", "kind",
    "seems", "seem", "might", "could", "apparently", "arguably", "roughly",
    "likely", "guess", "suppose", "basically", "actually", "just", "really",
}
BE_FORMS = {"is", "are", "was", "were", "be", "been", "being", "am"}
IRREGULAR_PARTICIPLES = {"made", "done", "seen", "taken", "given", "written", "known", "built", "found", "held", "shown", "chosen"}

# QWERTY layout, with finger assignment (0-3 left pinky..index, 4-7 right index..pinky)
KEY_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
FINGER = {}
for row, fingers in zip(
    KEY_ROWS,
    [
        [0, 1, 2, 3, 3, 4, 4, 5, 6, 7],
        [0, 1, 2, 3, 3, 4, 4, 5, 6],
        [0, 1, 2, 3, 3, 4, 4],
    ],
):
    for ch, f in zip(row, fingers):
        FINGER[ch] = f
HAND = {ch: ("L" if f <= 3 else "R") for ch, f in FINGER.items()}
HOME_ROW = set("asdfghjkl")

SAMPLES = {
    "Blog-style rambling": (
        "So I finally tried that new coffee place downtown... and honestly? It was amazing!! "
        "The barista was super friendly, the latte was perfectly smooth, and I couldn't stop smiling. "
        "Maybe it's just me, but I think good coffee makes everything better. "
        "Anyway, I'm definitely going back tomorrow. Who wants to join me? "
        "We could try the pastries too, they looked sooooo good."
    ),
    "Formal / academic": (
        "The relationship between sleep duration and cognitive performance has been extensively studied. "
        "However, the mechanisms underlying this association remain poorly understood. "
        "Previous research suggests that sleep deprivation impairs attention, memory consolidation, and executive function; "
        "furthermore, these effects appear to be cumulative over successive nights. "
        "Consequently, interventions that were designed to improve sleep hygiene have been proposed as a cost-effective strategy. "
        "The data were analyzed using mixed-effects models, and the results are presented below."
    ),
    "Literary prose": (
        "The house stood at the end of the lane, grey and patient. "
        "Rain had been falling for three days, and the garden, once so carefully tended, had surrendered to weeds. "
        "She stood at the window. Watching. Waiting. "
        "There was a time when the sound of the gate would have made her heart leap, but now it was only the wind, "
        "rattling the hinges as it had done every evening since he left."
    ),
    "Quick chat messages": (
        "hey are you free later?? i was thinking we could grab food. "
        "LOL no way thats actually happening. "
        "ok ok ok but seriously, where do u wanna go. "
        "pizza?? tacos?? idk im so hungry rn!!! "
        "lemme know asap, dont leave me hanging"
    ),
}

# ----------------------------------------------------------------------------
# Analysis helpers
# ----------------------------------------------------------------------------
ABBREVIATIONS = r"(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e|U\.S|No)"


def split_sentences(text: str) -> list[str]:
    """Heuristic sentence splitter that respects common abbreviations."""
    protected = re.sub(rf"\b({ABBREVIATIONS})\.", r"\1<DOT>", text)
    parts = re.split(r"(?<=[.!?…])[\"')\]]*\s+(?=[\"'(\[]?[A-Za-z0-9])", protected.strip())
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)


def count_syllables(word: str) -> int:
    word = word.lower()
    word = re.sub(r"[^a-z]", "", word)
    if not word:
        return 0
    if len(word) <= 3:
        return 1
    word = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", word)
    word = re.sub(r"^y", "", word)
    return max(1, len(re.findall(r"[aeiouy]{1,2}", word)))


def classify_sentence(s: str) -> str:
    stripped = s.rstrip("\"')] ")
    if stripped.endswith("?"):
        return "Question"
    if stripped.endswith("!"):
        return "Exclamation"
    words = tokenize_words(s)
    if len(words) <= 3:
        return "Fragment"
    return "Statement"


def structure_of(s: str) -> str:
    """Very rough clause-structure heuristic based on punctuation and conjunctions."""
    low = s.lower()
    subordinators = r"\b(although|though|because|since|while|whereas|unless|if|when|whenever|after|before|until|which|that|who)\b"
    coordinators = r"(,\s*(and|but|or|so|yet)\b|;)"
    has_sub = bool(re.search(subordinators, low))
    has_coord = bool(re.search(coordinators, low))
    if has_sub and has_coord:
        return "Compound-complex"
    if has_coord:
        return "Compound"
    if has_sub and len(tokenize_words(s)) > 8:
        return "Complex"
    return "Simple"


def is_passive(s: str) -> bool:
    words = [w.lower() for w in tokenize_words(s)]
    for i, w in enumerate(words[:-1]):
        if w in BE_FORMS:
            nxt = words[i + 1]
            if nxt in IRREGULAR_PARTICIPLES or (nxt.endswith("ed") and len(nxt) > 3):
                return True
            if i + 2 < len(words) and words[i + 1] in {"been", "being"}:
                n2 = words[i + 2]
                if n2.endswith("ed") or n2 in IRREGULAR_PARTICIPLES:
                    return True
    return False


def flesch_reading_ease(words: list[str], n_sent: int) -> float:
    if not words or n_sent == 0:
        return 0.0
    syl = sum(count_syllables(w) for w in words)
    return 206.835 - 1.015 * (len(words) / n_sent) - 84.6 * (syl / len(words))


def grade_level(words: list[str], n_sent: int) -> float:
    if not words or n_sent == 0:
        return 0.0
    syl = sum(count_syllables(w) for w in words)
    return 0.39 * (len(words) / n_sent) + 11.8 * (syl / len(words)) - 15.59


def ease_label(score: float) -> str:
    if score >= 90:
        return "Very easy"
    if score >= 70:
        return "Easy"
    if score >= 60:
        return "Plain English"
    if score >= 50:
        return "Fairly difficult"
    if score >= 30:
        return "Difficult"
    return "Very difficult"


def analyze(text: str) -> dict:
    sentences = split_sentences(text)
    words = tokenize_words(text)
    lower_words = [w.lower() for w in words]
    letters = [c for c in text.lower() if c.isalpha()]

    # --- sentence-level table
    rows = []
    for i, s in enumerate(sentences, 1):
        sw = tokenize_words(s)
        rows.append(
            {
                "#": i,
                "Sentence": s,
                "Words": len(sw),
                "Chars": len(s),
                "Avg word len": round(sum(len(w) for w in sw) / len(sw), 2) if sw else 0,
                "Type": classify_sentence(s),
                "Structure": structure_of(s),
                "Passive": is_passive(s),
                "Commas": s.count(","),
                "Opener": sw[0].lower() if sw else "",
            }
        )
    sent_df = pd.DataFrame(rows)

    # --- typing patterns
    letter_counts = Counter(letters)
    hand_counts = Counter(HAND[c] for c in letters if c in HAND)
    home_row = sum(1 for c in letters if c in HOME_ROW)
    letters_in_layout = sum(1 for c in letters if c in FINGER)

    alternations, same_finger, bigrams_total = 0, 0, 0
    for w in lower_words:
        w = re.sub(r"[^a-z]", "", w)
        for a, b in zip(w, w[1:]):
            if a in HAND and b in HAND:
                bigrams_total += 1
                if HAND[a] != HAND[b]:
                    alternations += 1
                elif FINGER[a] == FINGER[b] and a != b:
                    same_finger += 1

    punct = Counter(c for c in text if c in ".,!?;:-—–'\"()…")
    punct["…"] += text.count("...")
    punct["."] -= text.count("...") * 3  # ellipses were counted as 3 periods
    punct["."] = max(punct["."], 0)

    all_caps_words = [w for w in words if len(w) > 1 and w.isupper()]
    elongated = re.findall(r"\b\w*([a-zA-Z])\1{2,}\w*\b", text)
    multi_punct = re.findall(r"[!?]{2,}", text)
    double_spaces = len(re.findall(r"(?<=\S)  +(?=\S)", text))
    emoji = re.findall(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", text)
    lowercase_starts = sum(1 for s in sentences if s and s[0].islower())
    contractions = re.findall(r"\b\w+'(?:t|s|re|ve|ll|d|m)\b", text, flags=re.I)
    no_apostrophe = re.findall(r"\b(dont|cant|wont|im|ive|thats|isnt|didnt|doesnt|youre|theyre)\b", text, flags=re.I)

    # --- vocabulary
    content_words = [w for w in lower_words if w not in STOPWORDS and len(w) > 2]
    freq = Counter(content_words)
    ttr = len(set(lower_words)) / len(lower_words) if lower_words else 0
    # Root TTR is less length-sensitive than raw TTR
    rttr = len(set(lower_words)) / math.sqrt(len(lower_words)) if lower_words else 0
    hapax = sum(1 for _, c in Counter(lower_words).items() if c == 1)
    word_lengths = Counter(len(w) for w in words)

    n_sent = max(len(sentences), 1)
    return {
        "text": text,
        "sentences": sentences,
        "sent_df": sent_df,
        "words": words,
        "lower_words": lower_words,
        "letters": letters,
        "letter_counts": letter_counts,
        "hand_counts": hand_counts,
        "home_row_pct": home_row / letters_in_layout if letters_in_layout else 0,
        "alternation_rate": alternations / bigrams_total if bigrams_total else 0,
        "same_finger_rate": same_finger / bigrams_total if bigrams_total else 0,
        "punct": punct,
        "all_caps": all_caps_words,
        "elongated": len(elongated),
        "multi_punct": len(multi_punct),
        "double_spaces": double_spaces,
        "emoji": len(emoji),
        "lowercase_starts": lowercase_starts,
        "contractions": len(contractions),
        "no_apostrophe": len(no_apostrophe),
        "freq": freq,
        "ttr": ttr,
        "rttr": rttr,
        "hapax": hapax,
        "word_lengths": word_lengths,
        "flesch": flesch_reading_ease(words, n_sent),
        "grade": grade_level(words, n_sent),
        "transitions": sum(1 for w in lower_words if w in TRANSITIONS),
        "hedges": sum(1 for w in lower_words if w in HEDGES),
    }


def style_profile(a: dict) -> dict:
    """Map raw stats onto 0-100 axes for the radar chart."""
    df = a["sent_df"]
    n_words = max(len(a["words"]), 1)
    n_sent = max(len(df), 1)

    def clamp(x):
        return max(0, min(100, x))

    avg_len = df["Words"].mean() if not df.empty else 0
    len_std = df["Words"].std(ddof=0) if len(df) > 1 else 0
    complex_share = df["Structure"].isin(["Complex", "Compound", "Compound-complex"]).mean() if not df.empty else 0
    informal = (
        a["elongated"] + a["multi_punct"] + a["emoji"] + len(a["all_caps"]) + a["lowercase_starts"] + a["no_apostrophe"]
    ) / n_sent
    punct_total = sum(a["punct"].values())
    return {
        "Sentence length": clamp(avg_len / 30 * 100),
        "Rhythm variety": clamp(len_std / 12 * 100),
        "Structural complexity": clamp(complex_share * 100),
        "Vocabulary richness": clamp(a["rttr"] / 8 * 100),
        "Formality": clamp(100 - informal * 40 - (a["contractions"] / n_sent) * 10),
        "Expressiveness": clamp((a["punct"].get("!", 0) + a["punct"].get("?", 0) + a["emoji"]) / n_sent * 40),
        "Punctuation density": clamp(punct_total / n_words * 250),
    }


def persona(a: dict, profile: dict) -> tuple[str, str]:
    if profile["Formality"] < 45 and profile["Expressiveness"] > 40:
        return "🎉 The Enthusiastic Texter", "Casual, expressive, and quick on the keys. Punctuation is used for emotion, not just grammar."
    if profile["Formality"] < 45:
        return "💬 The Casual Conversationalist", "Relaxed and informal, with short natural phrasing that reads like speech."
    if profile["Structural complexity"] > 55 and profile["Sentence length"] > 50:
        return "🎓 The Academic Architect", "Long, layered sentences with subordinate clauses and careful connectors."
    if profile["Rhythm variety"] > 55:
        return "🎼 The Rhythmic Storyteller", "Varies sentence length deliberately, mixing long flowing lines with short punchy ones."
    if profile["Sentence length"] < 35:
        return "⚡ The Minimalist", "Crisp, direct sentences that get to the point."
    return "📝 The Balanced Communicator", "A steady, readable mix of sentence lengths and structures."


# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------
PALETTE = ["#6C5CE7", "#00B894", "#FD79A8", "#FDCB6E", "#0984E3", "#E17055", "#A29BFE"]


def style_fig(fig, height=340):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(size=13),
        legend=dict(orientation="h", y=-0.2),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def sentence_rhythm_chart(df):
    fig = go.Figure()
    fig.add_bar(x=df["#"], y=df["Words"], name="Words", marker_color=PALETTE[0], opacity=0.75)
    if len(df) >= 3:
        fig.add_scatter(
            x=df["#"], y=df["Words"].rolling(3, center=True, min_periods=1).mean(),
            name="3-sentence average", mode="lines", line=dict(color=PALETTE[2], width=3),
        )
    fig.update_layout(title="Sentence rhythm (words per sentence)", xaxis_title="Sentence #", yaxis_title="Words")
    return style_fig(fig)


def donut(counter_like: dict, title: str):
    labels, values = list(counter_like.keys()), list(counter_like.values())
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.55, marker=dict(colors=PALETTE), textinfo="label+percent"))
    fig.update_layout(title=title, showlegend=False)
    return style_fig(fig)


def keyboard_heatmap(letter_counts: Counter):
    total = sum(letter_counts.values()) or 1
    offsets = [0, 0.5, 1.0]
    max_len = 10
    z, text = [], []
    for row, off in zip(KEY_ROWS, offsets):
        zr = [None] * max_len
        tr = [""] * max_len
        for i, ch in enumerate(row):
            zr[i] = letter_counts.get(ch, 0) / total * 100
            tr[i] = f"{ch.upper()}<br>{zr[i]:.1f}%"
        z.append(zr)
        text.append(tr)
    fig = go.Figure(
        go.Heatmap(
            z=z, text=text, texttemplate="%{text}", colorscale="Purples",
            showscale=False, xgap=4, ygap=4, hoverinfo="skip",
        )
    )
    fig.update_yaxes(autorange="reversed", visible=False)
    fig.update_xaxes(visible=False)
    fig.update_layout(title="Keyboard heatmap (share of letters typed)")
    return style_fig(fig, 280)


def finger_chart(letter_counts: Counter):
    names = ["L pinky", "L ring", "L middle", "L index", "R index", "R middle", "R ring", "R pinky"]
    counts = [0] * 8
    for ch, n in letter_counts.items():
        if ch in FINGER:
            counts[FINGER[ch]] += n
    fig = go.Figure(
        go.Bar(x=names, y=counts, marker_color=[PALETTE[4]] * 4 + [PALETTE[2]] * 4)
    )
    fig.update_layout(title="Finger workload", yaxis_title="Keystrokes")
    return style_fig(fig, 280)


def radar(profile: dict):
    cats = list(profile.keys())
    vals = list(profile.values())
    fig = go.Figure(
        go.Scatterpolar(
            r=vals + vals[:1], theta=cats + cats[:1], fill="toself",
            line=dict(color=PALETTE[0], width=3), fillcolor="rgba(108,92,231,0.30)",
        )
    )
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, title="Style fingerprint")
    return style_fig(fig, 420)


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .big-persona {font-size: 1.7rem; font-weight: 700; margin-bottom: 0.2rem;}
    div[data-testid="stMetric"] {
        background: rgba(108,92,231,0.08); border-radius: 12px; padding: 12px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧬 Writing DNA")
st.caption("Paste any text to explore how it was typed and how its sentences are built.")

with st.sidebar:
    st.header("Input")
    sample = st.selectbox("Load a sample", ["(none)"] + list(SAMPLES.keys()))
    upload = st.file_uploader("...or upload a .txt file", type=["txt", "md"])
    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "Everything runs locally with simple heuristics, so results are indicative, not linguistic gospel. "
        "Keyboard stats assume a QWERTY layout and touch typing."
    )

default_text = SAMPLES.get(sample, "") if sample != "(none)" else ""
if upload is not None:
    default_text = upload.read().decode("utf-8", errors="ignore")

text = st.text_area("Your text", value=default_text, height=200, placeholder="Paste or type something (at least a few sentences works best)...")

if len(tokenize_words(text)) < 5:
    st.info("Enter at least a few words to generate the dashboard. Try a sample from the sidebar.")
    st.stop()

a = analyze(text)
df = a["sent_df"]
profile = style_profile(a)
title, blurb = persona(a, profile)

# --- Persona + headline metrics
st.markdown(f'<div class="big-persona">{title}</div>', unsafe_allow_html=True)
st.write(blurb)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Words", f"{len(a['words']):,}")
m2.metric("Sentences", len(df))
m3.metric("Avg sentence", f"{df['Words'].mean():.1f} words")
m4.metric("Reading ease", f"{a['flesch']:.0f}", ease_label(a["flesch"]), delta_color="off")
m5.metric("Grade level", f"{max(a['grade'], 0):.1f}")

tab_over, tab_sent, tab_type, tab_vocab, tab_table = st.tabs(
    ["🧭 Overview", "🏗️ Sentence construction", "⌨️ Typing patterns", "📚 Vocabulary", "🔎 Sentence explorer"]
)

# ------------------------------- Overview
with tab_over:
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.plotly_chart(radar(profile), width="stretch")
    with c2:
        st.subheader("What stands out")
        notes = []
        longest = df.loc[df["Words"].idxmax()]
        shortest = df.loc[df["Words"].idxmin()]
        notes.append(f"Longest sentence is **#{int(longest['#'])}** at **{int(longest['Words'])} words**; shortest is **#{int(shortest['#'])}** at **{int(shortest['Words'])}**.")
        passive_pct = df["Passive"].mean() * 100
        notes.append(f"**{passive_pct:.0f}%** of sentences look passive.")
        if a["hedges"]:
            notes.append(f"Found **{a['hedges']}** hedging words (maybe, just, really...), which soften claims.")
        if a["transitions"]:
            notes.append(f"**{a['transitions']}** connectors/transition words help link ideas.")
        top_opener = df["Opener"].value_counts()
        if len(top_opener) and top_opener.iloc[0] > 1:
            notes.append(f"You open sentences with \"**{top_opener.index[0]}**\" {top_opener.iloc[0]} times.")
        if a["alternation_rate"]:
            notes.append(f"**{a['alternation_rate']*100:.0f}%** of letter pairs alternate hands, which feels smooth to type.")
        if a["elongated"] or a["multi_punct"] or a["emoji"]:
            notes.append(f"Expressive habits: {a['elongated']} stretched words, {a['multi_punct']} repeated !/?, {a['emoji']} emoji.")
        for n in notes:
            st.markdown(f"- {n}")

        st.subheader("Style scores")
        for k, v in profile.items():
            st.progress(int(v), text=f"{k}: {v:.0f}/100")

# ------------------------------- Sentence construction
with tab_sent:
    st.plotly_chart(sentence_rhythm_chart(df), width="stretch")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(donut(df["Type"].value_counts().to_dict(), "Sentence types"), width="stretch")
    with c2:
        order = ["Simple", "Compound", "Complex", "Compound-complex"]
        counts = df["Structure"].value_counts().reindex(order).fillna(0).astype(int)
        st.plotly_chart(donut(counts[counts > 0].to_dict(), "Clause structure"), width="stretch")
    with c3:
        fig = px.histogram(df, x="Words", nbins=min(15, max(df["Words"].nunique(), 3)), color_discrete_sequence=[PALETTE[1]])
        fig.update_layout(title="Length distribution", yaxis_title="Sentences")
        st.plotly_chart(style_fig(fig), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        op = df["Opener"].value_counts().head(8).reset_index()
        op.columns = ["Opener", "Count"]
        fig = px.bar(op, x="Count", y="Opener", orientation="h", color_discrete_sequence=[PALETTE[0]])
        fig.update_layout(title="Most common sentence openers", yaxis=dict(autorange="reversed"))
        st.plotly_chart(style_fig(fig), width="stretch")
    with c2:
        fig = px.scatter(
            df, x="Words", y="Commas", size="Chars", color="Structure", hover_data=["Sentence"],
            color_discrete_sequence=PALETTE,
        )
        fig.update_layout(title="Length vs. comma use (bubble = characters)")
        st.plotly_chart(style_fig(fig), width="stretch")

    # Rhythm insight
    std = df["Words"].std(ddof=0) if len(df) > 1 else 0
    mean = df["Words"].mean()
    cv = std / mean if mean else 0
    if cv > 0.6:
        st.success("🎼 **Highly varied rhythm.** Your sentence lengths swing widely, which keeps prose lively.")
    elif cv > 0.3:
        st.info("🎵 **Moderate rhythm.** A comfortable mix of longer and shorter sentences.")
    else:
        st.warning("🥁 **Very uniform rhythm.** Sentences are similar in length. Try mixing in a short punchy one.")

# ------------------------------- Typing patterns
with tab_type:
    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.plotly_chart(keyboard_heatmap(a["letter_counts"]), width="stretch")
    with c2:
        hc = a["hand_counts"]
        total_h = sum(hc.values()) or 1
        st.plotly_chart(
            donut({"Left hand": hc.get("L", 0), "Right hand": hc.get("R", 0)}, "Hand balance"),
            width="stretch",
        )

    st.plotly_chart(finger_chart(a["letter_counts"]), width="stretch")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Home-row letters", f"{a['home_row_pct']*100:.0f}%")
    k2.metric("Hand alternation", f"{a['alternation_rate']*100:.0f}%", help="Share of consecutive letter pairs typed with opposite hands. Higher usually means faster flow.")
    k3.metric("Same-finger pairs", f"{a['same_finger_rate']*100:.1f}%", help="Different keys in a row hit by the same finger. These are the slow ones.")
    k4.metric("Est. typing effort", f"{(a['same_finger_rate']*3 + (1 - a['alternation_rate'])) * 50:.0f}/100", help="Rough index: lower is smoother.")

    st.subheader("Punctuation & habits")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        pdf = pd.DataFrame({"Mark": list(a["punct"].keys()), "Count": list(a["punct"].values())})
        pdf = pdf[pdf["Count"] > 0].sort_values("Count", ascending=False)
        fig = px.bar(pdf, x="Mark", y="Count", color_discrete_sequence=[PALETTE[2]])
        fig.update_layout(title="Punctuation usage")
        st.plotly_chart(style_fig(fig, 300), width="stretch")
    with c2:
        habits = pd.DataFrame(
            {
                "Habit": [
                    "ALL-CAPS words", "Stretched letters (sooo)", "Repeated !!/??", "Double spaces",
                    "Emoji", "Lowercase sentence starts", "Contractions", "Missing apostrophes (dont)",
                ],
                "Count": [
                    len(a["all_caps"]), a["elongated"], a["multi_punct"], a["double_spaces"],
                    a["emoji"], a["lowercase_starts"], a["contractions"], a["no_apostrophe"],
                ],
            }
        )
        st.dataframe(habits, hide_index=True, width="stretch")

# ------------------------------- Vocabulary
with tab_vocab:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Unique words", len(set(a["lower_words"])))
    c2.metric("Type-token ratio", f"{a['ttr']:.2f}")
    c3.metric("Root TTR", f"{a['rttr']:.1f}", help="Length-adjusted lexical diversity. Above ~7 is rich.")
    c4.metric("Words used once", f"{a['hapax']}")

    c1, c2 = st.columns(2)
    with c1:
        top = pd.DataFrame(a["freq"].most_common(12), columns=["Word", "Count"])
        if not top.empty:
            fig = px.bar(top, x="Count", y="Word", orientation="h", color_discrete_sequence=[PALETTE[4]])
            fig.update_layout(title="Top content words (stopwords removed)", yaxis=dict(autorange="reversed"))
            st.plotly_chart(style_fig(fig, 400), width="stretch")
    with c2:
        wl = pd.DataFrame(sorted(a["word_lengths"].items()), columns=["Length", "Count"])
        fig = px.bar(wl, x="Length", y="Count", color_discrete_sequence=[PALETTE[3]])
        fig.update_layout(title="Word length distribution", xaxis_title="Letters per word")
        st.plotly_chart(style_fig(fig, 400), width="stretch")

    long_words = sorted({w.lower() for w in a["words"] if len(w) >= 9}, key=len, reverse=True)[:15]
    if long_words:
        st.markdown("**Longest words:** " + ", ".join(f"`{w}`" for w in long_words))

# ------------------------------- Sentence explorer
with tab_table:
    st.caption("Sort, filter, and inspect every sentence.")
    f1, f2, f3 = st.columns(3)
    types = f1.multiselect("Type", sorted(df["Type"].unique()), default=list(df["Type"].unique()))
    structs = f2.multiselect("Structure", sorted(df["Structure"].unique()), default=list(df["Structure"].unique()))
    min_len = f3.slider("Min words", 0, int(df["Words"].max()), 0)
    view = df[df["Type"].isin(types) & df["Structure"].isin(structs) & (df["Words"] >= min_len)]
    st.dataframe(
        view.drop(columns=["Opener"]),
        hide_index=True,
        width="stretch",
        column_config={
            "Sentence": st.column_config.TextColumn(width="large"),
            "Words": st.column_config.ProgressColumn(min_value=0, max_value=int(df["Words"].max()), format="%d"),
        },
    )
    st.download_button("Download analysis (CSV)", df.to_csv(index=False), "sentence_analysis.csv", "text/csv")
