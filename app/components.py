"""Reusable UI components and async generation job for ForgeFlow AI."""

import io
import threading
import logging

import streamlit as st
import polars as pl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Async generation job with stop support
# ---------------------------------------------------------------------------
class GenerationJob:
    """Background generation job with progress tracking and cancellation."""

    def __init__(self):
        self.status: str = "idle"          # idle | running | complete | stopped | error
        self.progress: float = 0.0
        self.status_text: str = ""
        self.records_done: int = 0
        self.total_records: int = 0
        self.result_df: pl.DataFrame | None = None
        self.error_msg: str | None = None
        self._stop = threading.Event()

    # -- public API --
    def start(self, generate_fn, *args, **kwargs):
        """Launch *generate_fn* in a daemon thread."""
        self.status = "running"
        self.progress = 0.0
        self.records_done = 0
        self.result_df = None
        self.error_msg = None
        self._stop.clear()

        kwargs["progress_callback"] = self._on_progress
        kwargs["stop_check"] = self._should_stop

        t = threading.Thread(target=self._run, args=(generate_fn, args, kwargs), daemon=True)
        t.start()

    def request_stop(self):
        self._stop.set()

    # -- internal --
    def _run(self, fn, args, kwargs):
        try:
            result = fn(*args, **kwargs)
            self.result_df = result
            if self._stop.is_set():
                self.status = "stopped"
                n = len(result) if result is not None else 0
                self.status_text = f"Stopped — {n:,} records generated"
            else:
                n = len(result) if result is not None else 0
                self.status = "complete"
                self.status_text = f"Generated {n:,} records"
        except Exception as e:
            self.status = "error"
            self.error_msg = str(e)
            self.status_text = f"Error: {e}"
            logger.exception("Generation job failed")

    def _on_progress(self, done: int, total: int):
        self.records_done = done
        self.total_records = total
        self.progress = done / total if total > 0 else 0
        self.status_text = f"Generating… {done:,} / {total:,} records"

    def _should_stop(self) -> bool:
        return self._stop.is_set()


# ---------------------------------------------------------------------------
# HTML component helpers
# ---------------------------------------------------------------------------
def hero(title: str, subtitle: str):
    """Render a gradient hero banner."""
    st.markdown(
        f'<div class="hero-banner"><h2>{title}</h2><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def status_pill(text: str, variant: str = "green"):
    """Render a coloured status pill.  variant: green | red | blue | amber"""
    st.markdown(
        f'<span class="status-pill pill-{variant}">{text}</span>',
        unsafe_allow_html=True,
    )


def metric_row(metrics: list[tuple[str, str]]):
    """Render a row of metric tiles.  metrics = [(value, label), ...]"""
    tiles = "".join(
        f'<div class="metric-tile"><div class="val">{v}</div><div class="lbl">{l}</div></div>'
        for v, l in metrics
    )
    st.markdown(f'<div class="metric-row">{tiles}</div>', unsafe_allow_html=True)


def empty_state(icon: str, message: str, hint: str = ""):
    """Render a friendly empty-state placeholder."""
    h = f'<div class="empty-state-hint">{hint}</div>' if hint else ""
    st.markdown(
        f'<div class="empty-state"><div class="icon">{icon}</div>'
        f'<div class="msg">{message}</div>{h}</div>',
        unsafe_allow_html=True,
    )


def section(icon: str, title: str):
    """Render a section header with icon."""
    st.markdown(f'<div class="section-hdr">{icon} {title}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def download_buttons(df: pl.DataFrame, prefix: str = "synthetic_data"):
    """Render CSV / JSON / Parquet / ZIP download buttons."""
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button(
            "📥  CSV",
            data=df.write_csv().encode("utf-8"),
            file_name=f"{prefix}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        buf = io.BytesIO()
        df.write_ndjson(buf)
        st.download_button(
            "📥  JSON",
            data=buf.getvalue(),
            file_name=f"{prefix}.ndjson",
            mime="application/x-ndjson",
            use_container_width=True,
        )
    with c3:
        buf = io.BytesIO()
        df.write_parquet(buf)
        st.download_button(
            "📥  Parquet",
            data=buf.getvalue(),
            file_name=f"{prefix}.parquet",
            mime="application/octet-stream",
            use_container_width=True,
        )
    with c4:
        import zipfile
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{prefix}.csv", df.write_csv())
            ndjson_buf = io.BytesIO()
            df.write_ndjson(ndjson_buf)
            zf.writestr(f"{prefix}.ndjson", ndjson_buf.getvalue())
            pq_buf = io.BytesIO()
            df.write_parquet(pq_buf)
            zf.writestr(f"{prefix}.parquet", pq_buf.getvalue())
        st.download_button(
            "📦  ZIP (all)",
            data=zip_buf.getvalue(),
            file_name=f"{prefix}.zip",
            mime="application/zip",
            use_container_width=True,
        )
