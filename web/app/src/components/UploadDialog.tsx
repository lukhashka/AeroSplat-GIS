import { useRef, useState, type DragEvent } from "react";
import { useNavigate } from "react-router-dom";

import { useSystem, useUploadCapture } from "../api/hooks";
import type { CaptureSource } from "../api/types";
import { SLUG_RE, bytes, slugify } from "../lib/format";
import { ErrorBox, Modal } from "./Modal";
import { SOURCE_LABEL } from "./StatusBadge";

const IMAGE_RE = /\.(jpe?g|png|tiff?)$/i;
const EXTRA_RE = /\.(json|esp|txt)$/i;

const SOURCE_HELP: Record<CaptureSource, string> = {
  earthstudio: "Експорт Google Earth Studio: файл .json з 3D-трекінгом і всі кадри (_00.jpeg …). Пози камер точні.",
  drone: "Фото з дрона. GPS з EXIF використаємо як пріор (реєстрація — наступні етапи).",
  handheld: "Фото з телефона чи камери навколо об'єкта. Знімай з перекриттям ~70%, з двох висот.",
  internet: "Фото з інтернету. Обов'язково вкажи ліцензію і автора кожної партії.",
};

export function UploadDialog({
  project,
  site,
  existing,
  onClose,
}: {
  project: string;
  site: string;
  existing: string[];
  onClose: () => void;
}) {
  const uploadMut = useUploadCapture(project, site);
  const navigate = useNavigate();
  const { data: sys } = useSystem();
  const cpuOnline = !sys || sys.workers.some((w) => w.resource === "cpu" && w.online);
  const input = useRef<HTMLInputElement>(null);
  const [source, setSource] = useState<CaptureSource>("handheld");
  const [files, setFiles] = useState<File[]>([]);
  const [captureId, setCaptureId] = useState("");
  const [license, setLicense] = useState("");
  const [author, setAuthor] = useState("");
  const [over, setOver] = useState(false);
  const [progress, setProgress] = useState(0);

  const images = files.filter((f) => IMAGE_RE.test(f.name));
  const extras = files.filter((f) => EXTRA_RE.test(f.name));
  const accepted = [...images, ...(source === "earthstudio" ? extras : [])];
  const total = accepted.reduce((a, f) => a + f.size, 0);
  const jsonCount = extras.filter((f) => f.name.toLowerCase().endsWith(".json")).length;
  const defaultId = `${source}-${new Date().toISOString().slice(0, 10)}`;
  const id = captureId || defaultId;

  const problem =
    !images.length
      ? "додай зображення"
      : source === "earthstudio" && jsonCount !== 1
        ? "для Earth Studio потрібен рівно один .json експорту"
        : !SLUG_RE.test(id)
          ? "id захоплення: малі латинські літери, цифри, . _ -"
          : existing.includes(id)
            ? `захоплення «${id}» вже існує`
            : source === "internet" && !license.trim()
              ? "для фото з інтернету вкажи ліцензію"
              : null;

  const addFiles = (list: FileList | null) => {
    if (!list) return;
    const incoming = Array.from(list).filter((f) => IMAGE_RE.test(f.name) || EXTRA_RE.test(f.name));
    setFiles((prev) => {
      const names = new Set(prev.map((f) => f.name));
      return [...prev, ...incoming.filter((f) => !names.has(f.name))];
    });
    if (incoming.some((f) => f.name.toLowerCase().endsWith(".json"))) setSource("earthstudio");
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setOver(false);
    addFiles(e.dataTransfer.files);
  };

  const submit = () => {
    if (problem) return;
    uploadMut.mutate(
      { captureId: id, source, license, author, files: accepted, onProgress: setProgress },
      {
        // The files are on the server; hashing and registration are a job on the CPU worker.
        // Its page shows the progress; the capture appears on the site when the job succeeds.
        onSuccess: (job) => {
          onClose();
          navigate(`/jobs/${job.id}`);
        },
      },
    );
  };

  return (
    <Modal
      title="Нове захоплення"
      subtitle="Партія кадрів одного джерела з однаковою ліцензією й умовами зйомки."
      onClose={uploadMut.isPending ? () => undefined : onClose}
      footer={
        <>
          {uploadMut.isPending ? (
            <div className="grow" style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div className="progress grow">
                <i style={{ width: `${Math.round(progress * 100)}%` }} />
              </div>
              <span className="num muted" style={{ fontSize: 12 }}>
                {progress < 1 ? `${Math.round(progress * 100)}%` : "постановка в чергу…"}
              </span>
            </div>
          ) : (
            problem && <span className="faint grow" style={{ fontSize: 12 }}>{problem}</span>
          )}
          <button className="btn ghost" onClick={onClose} disabled={uploadMut.isPending}>
            Скасувати
          </button>
          <button className="btn primary" onClick={submit} disabled={!!problem || uploadMut.isPending}>
            Завантажити {accepted.length ? `(${accepted.length})` : ""}
          </button>
        </>
      }
    >
      <div className="chips">
        {(Object.keys(SOURCE_LABEL) as CaptureSource[]).map((s) => (
          <span key={s} className={`chip${source === s ? " on" : ""}`} onClick={() => setSource(s)}>
            {SOURCE_LABEL[s]}
          </span>
        ))}
      </div>
      <div className="note">{SOURCE_HELP[source]}</div>

      <div
        className={`dropzone${over ? " over" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={onDrop}
        onClick={() => input.current?.click()}
      >
        <input
          ref={input}
          type="file"
          multiple
          hidden
          accept=".jpg,.jpeg,.png,.tif,.tiff,.json,.esp,.txt"
          onChange={(e) => addFiles(e.target.files)}
        />
        {files.length ? (
          <>
            <b>{images.length}</b> зображень
            {source === "earthstudio" && (
              <>
                {" "}
                · <b>{extras.length}</b> файлів експорту
              </>
            )}{" "}
            · {bytes(total)}
            <div className="faint" style={{ fontSize: 12, marginTop: 4 }}>
              перетягни ще або клікни, щоб додати
            </div>
          </>
        ) : (
          <>
            <b>Перетягни сюди кадри</b> або клікни, щоб вибрати
            <div className="faint" style={{ fontSize: 12, marginTop: 4 }}>
              JPEG / PNG / TIFF{source === "earthstudio" ? " + .json експорту" : ""}
            </div>
          </>
        )}
      </div>

      <div className="form-grid">
        <label className="field">
          <span>Id захоплення</span>
          <input value={captureId} placeholder={defaultId} onChange={(e) => setCaptureId(slugify(e.target.value))} />
        </label>
        <label className="field">
          <span>Автор</span>
          <input value={author} onChange={(e) => setAuthor(e.target.value)} placeholder="хто знімав" />
        </label>
        {source !== "earthstudio" && (
          <label className="field full">
            <span>Ліцензія{source === "internet" ? " *" : ""}</span>
            <input value={license} onChange={(e) => setLicense(e.target.value)} placeholder="напр. CC-BY-SA-4.0 або «власна зйомка»" />
          </label>
        )}
      </div>
      {!cpuOnline && (
        <div className="note">
          CPU-воркер офлайн: файли завантажаться, але захоплення з'явиться лише після того, як воркер обробить задачу
          (<code>aerosplat up</code> або <code>aerosplat worker --resource cpu</code>).
        </div>
      )}
      <ErrorBox error={uploadMut.error} />
    </Modal>
  );
}
