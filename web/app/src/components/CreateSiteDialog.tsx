import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

import { useCreateSite, useProjects } from "../api/hooks";
import type { SiteKind } from "../api/types";
import { SLUG_RE, coord, slugify } from "../lib/format";
import { ErrorBox, Modal } from "./Modal";
import { SiteMap, type MapPoint } from "./SiteMap";
import { KIND_LABEL } from "./StatusBadge";

const NEW = "__new__";

export function CreateSiteDialog({ onClose }: { onClose: () => void }) {
  const navigate = useNavigate();
  const { data: projects = [] } = useProjects();
  const create = useCreateSite();
  const [projectSel, setProjectSel] = useState<string>("");
  const [newProject, setNewProject] = useState("");
  const [name, setName] = useState("");
  const [id, setId] = useState("");
  const [idTouched, setIdTouched] = useState(false);
  const [kind, setKind] = useState<SiteKind>("urban");
  const [pick, setPick] = useState<MapPoint | null>(null);
  const [h, setH] = useState("0");

  const isNewProject = !projects.length || projectSel === NEW;
  const project = isNewProject ? newProject : projectSel || projects[0]!.id;
  const siteId = idTouched ? id : slugify(name);
  const height = Number(h);

  const problems = useMemo(() => {
    const p: string[] = [];
    if (!SLUG_RE.test(project)) p.push("id проєкту: малі латинські літери, цифри, . _ -");
    if (!SLUG_RE.test(siteId)) p.push("id ділянки: малі латинські літери, цифри, . _ -");
    if (!pick) p.push("клікни на карті, щоб задати початок системи координат");
    if (!Number.isFinite(height)) p.push("висота має бути числом");
    return p;
  }, [project, siteId, pick, height]);

  const submit = () => {
    if (problems.length || !pick) return;
    create.mutate(
      { project, newProject: isNewProject, id: siteId, name, kind, origin: { lat: pick.lat, lon: pick.lon, h: height } },
      { onSuccess: (site) => navigate(`/sites/${site.project_id}/${site.id}`) },
    );
  };

  return (
    <Modal
      wide
      title="Нова ділянка"
      subtitle="Ділянка — територія до ~1 км² з власною метричною системою координат. Її початок фіксується назавжди."
      onClose={onClose}
      footer={
        <>
          {problems.length > 0 && <span className="faint grow" style={{ fontSize: 12 }}>{problems[0]}</span>}
          <button className="btn ghost" onClick={onClose}>
            Скасувати
          </button>
          <button className="btn primary" disabled={problems.length > 0 || create.isPending} onClick={submit}>
            {create.isPending ? "Створення…" : "Створити ділянку"}
          </button>
        </>
      }
    >
      <SiteMap
        pick={pick}
        onPick={setPick}
        hint={pick ? `Початок: ${coord(pick.lat, pick.lon)}` : "Клікни в центр ділянки — це буде ENU-початок"}
      />
      <div className="form-grid">
        <label className="field">
          <span>Проєкт</span>
          {projects.length > 0 ? (
            <select value={projectSel || projects[0]!.id} onChange={(e) => setProjectSel(e.target.value)}>
              {projects.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name || p.id}
                </option>
              ))}
              <option value={NEW}>+ Новий проєкт…</option>
            </select>
          ) : (
            <input value={newProject} onChange={(e) => setNewProject(slugify(e.target.value))} placeholder="напр. food-delivery" />
          )}
        </label>
        {projects.length > 0 && projectSel === NEW ? (
          <label className="field">
            <span>Id нового проєкту</span>
            <input value={newProject} onChange={(e) => setNewProject(slugify(e.target.value))} placeholder="напр. flood-response" autoFocus />
          </label>
        ) : (
          <div />
        )}
        <label className="field">
          <span>Назва</span>
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="напр. Склад на Подолі" />
        </label>
        <label className="field">
          <span>Id ділянки</span>
          <input
            value={siteId}
            onChange={(e) => {
              setIdTouched(true);
              setId(e.target.value.toLowerCase());
            }}
            placeholder="podil-depot"
          />
          <small>використовується в шляхах і адресах; змінити потім не можна</small>
        </label>
        <label className="field">
          <span>Тип ділянки</span>
          <select value={kind} onChange={(e) => setKind(e.target.value as SiteKind)}>
            {Object.entries(KIND_LABEL).map(([k, v]) => (
              <option key={k} value={k}>
                {v}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span>Висота початку, м</span>
          <input value={h} onChange={(e) => setH(e.target.value)} inputMode="decimal" />
          <small>еліпсоїдальна (WGS84), не над рівнем моря; можна приблизно</small>
        </label>
      </div>
      <ErrorBox error={create.error} />
    </Modal>
  );
}
