import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from if_curator import embeddings, frigate
from if_curator.onnx_runtime import preload_cuda


@pytest.mark.parametrize("providers", [["CPUExecutionProvider"], ["CoreMLExecutionProvider"]])
def test_non_cuda_does_not_preload(providers):
    ort = SimpleNamespace(preload_dlls=Mock())
    preload_cuda(ort, providers)
    ort.preload_dlls.assert_not_called()


def test_insightface_preloads_before_constructing_sessions(monkeypatch):
    events = []
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    monkeypatch.delenv("FORCE_CPU", raising=False)
    monkeypatch.setattr(embeddings, "_insightface_app", None)
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            get_available_providers=lambda: providers,
            preload_dlls=lambda: events.append("preload"),
        ),
    )
    app = SimpleNamespace(
        prepare=Mock(), det_model=SimpleNamespace(session=SimpleNamespace(get_providers=lambda: providers))
    )

    def construct(**kwargs):
        events.append("session")
        return app

    monkeypatch.setitem(sys.modules, "insightface.app", SimpleNamespace(FaceAnalysis=construct))
    assert embeddings.get_insightface_app() is app
    assert events == ["preload", "session"]
    assert embeddings.get_insightface_app() is app
    assert events == ["preload", "session"]


@pytest.mark.parametrize("force_cpu", [False, True])
def test_frigate_preloads_before_constructing_sessions(tmp_path, monkeypatch, force_cpu):
    events = []
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def construct(path, providers):
        events.append("session")
        return SimpleNamespace(get_providers=lambda: providers)

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            __version__="test",
            get_available_providers=lambda: providers,
            preload_dlls=lambda: events.append("preload"),
            InferenceSession=construct,
        ),
    )
    monkeypatch.setattr(frigate, "ensure_model", lambda directory, name: directory / name)
    monkeypatch.setattr(frigate, "file_hash", lambda path: "test")
    monkeypatch.setattr(frigate.cv2.face, "createFacemarkLBF", lambda: SimpleNamespace(loadModel=Mock()))
    model = frigate.FrigateModel(tmp_path, force_cpu)
    assert events == (["session"] if force_cpu else ["preload", "session"])
    assert model.session.get_providers() == (["CPUExecutionProvider"] if force_cpu else providers)
