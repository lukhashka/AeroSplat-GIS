import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";

import { useSemantics } from "../api/hooks";
import type { Coverage, Facade } from "../api/types";
import { Icon } from "../components/Icon";
import { ErrorBox } from "../components/Modal";

const OCTANT_LABEL: Record<string, string> = {
  N: "північ",
  NE: "північ-схід",
  E: "схід",
  SE: "південь-схід",
  S: "південь",
  SW: "південь-захід",
  W: "захід",
  NW: "північ-захід",
};

const facadeName = (id: string) => id.split("/").pop() ?? id;
const octant = (id: string) => facadeName(id).replace(/\d+$/, "");

/** How much of the wall the capture resolved finely enough to read openings off (§5.3). */
function CoverageBar({ coverage }: { coverage: Coverage | null }) {
  if (!coverage) return <span className="faint">немає</span>;
  const covered = coverage.cells ? coverage.covered / coverage.cells : 0;
  const visible = coverage.cells ? coverage.visible / coverage.cells : 0;
  // Seen from the right angles, but never sharply enough: the geometry is fine, the
  // flight was too far away. That distinction is the whole point of two numbers.
  const tooCoarse = coverage.visible > 0 && coverage.covered === 0;
  const tone = covered > 0.5 ? "var(--green)" : covered > 0.1 ? "var(--amber)" : "var(--red)";
  return (
    <div className="cov-cell">
      <div className="cov-bar" title={`видно ${(100 * visible).toFixed(0)}%, з потрібною деталізацією ${(100 * covered).toFixed(0)}%`}>
        <i style={{ width: `${100 * visible}%`, background: "var(--border-strong)" }} />
        <i style={{ width: `${100 * covered}%`, background: tone }} />
      </div>
      <small className="faint num">
        {tooCoarse ? (
          <span style={{ color: "var(--amber)" }}>
видно {(100 * visible).toFixed(0)}%, лише {coverage.best_gsd_cm?.toFixed(0) ?? "?"} см/px
          </span>
        ) : (
          <>
            {(100 * covered).toFixed(0)}% ≤ {coverage.max_gsd_cm.toFixed(0)} см/px
            {coverage.best_gsd_cm != null && ` · найкраще ${coverage.best_gsd_cm.toFixed(0)}`}
          </>
        )}
      </small>
    </div>
  );
}

export function FacadesPage() {
  const { project = "", site = "" } = useParams();
  const navigate = useNavigate();
  // `?run=<id>` lists that run's facades; default: the newest run with semantics. Facade
  // ids are per run (the same wall is B08/SW3 in one run and B04/SW1 in the next).
  const [params, setParams] = useSearchParams();
  const runParam = params.get("run");
  const { data, isLoading, error } = useSemantics(project, site, runParam);

  if (isLoading) return <div className="page faint">Завантаження семантики…</div>;
  if (error || !data)
    return (
      <div className="page">
        <ErrorBox error={error ?? "семантики ще немає"} />
        <p className="muted">
          Фасади з'являються після стадії <code>semantics</code>. Їй потрібна щільна хмара, тож run має
          пройти <code>dense</code> → <code>georef</code> → <code>semantics</code>.
        </p>
        <Link to={`/sites/${project}/${site}`}>← До ділянки</Link>
      </div>
    );

  const byBuilding = new Map<string, Facade[]>();
  for (const f of data.facades) {
    const list = byBuilding.get(f.building_id) ?? [];
    list.push(f);
    byBuilding.set(f.building_id, list);
  }
  const buildings = [...byBuilding.keys()].sort();
  const features = Object.values(data.features_by_facade).reduce((a, b) => a + b, 0);

  return (
    <div className="page" style={{ maxWidth: 1100 }}>
      <div className="page-head">
        <div>
          <div className="crumbs">
            <Link to="/">Ділянки</Link> / <Link to={`/sites/${project}/${site}`}>{project}/{site}</Link> / Фасади
          </div>
          <h1>Фасади й точки доступу</h1>
          <div className="sub row" style={{ gap: 14 }}>
            {data.runs.length > 1 ? (
              <label className="row" style={{ gap: 6 }} title="Фасади, їхні id і ортофото належать одному run-у">
                <span>run</span>
                <select
                  className="mono"
                  value={data.run_id}
                  onChange={(e) => {
                    const next = e.target.value;
                    setParams(next === data.runs[0]?.id ? {} : { run: next });
                  }}
                >
                  {data.runs.map((r, i) => (
                    <option key={r.id} value={r.id}>
                      {r.id}
                      {i === 0 ? " (найновіший)" : ""}
                      {r.scene ? "" : " — без сцени"}
                    </option>
                  ))}
                </select>
              </label>
            ) : (
              <span className="mono">run {data.run_id}</span>
            )}
            <span>{data.buildings.length} будівель · {data.facades.length} фасадів</span>
            <span>{features} точок доступу</span>
          </div>
        </div>
      </div>

      {data.warnings.length > 0 && (
        <div className="error-box" style={{ borderColor: "color-mix(in srgb, var(--amber) 45%, var(--border))" }}>
          {data.warnings.map((w, i) => (
            <div key={i}>⚠️ {w}</div>
          ))}
        </div>
      )}
      {!data.scene && (
        <div className="error-box">
          У цього run-а немає власної навченої сцени, тож ортофото його фасадів не відрендерити (сцену
          іншого run-а не підставляємо: фасад належить своїй реконструкції). Фасади нижче можна дивитися,
          але малювати по них — ні.
        </div>
      )}

      {buildings.map((b) => {
        const rec = data.buildings.find((x) => x.id === b);
        return (
          <div className="panel" key={b}>
            <div className="panel-head">
              <b>{b}</b>
              {rec && (
                <span className="faint num">
                  {rec.height_m.toFixed(1)} м · {rec.area_m2.toFixed(0)} м² · дах {rec.roof_z.toFixed(1)} м
                </span>
              )}
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Фасад</th>
                    <th>Напрямок</th>
                    <th className="num">Довжина</th>
                    <th className="num">Висота</th>
                    <th className="num">Площина</th>
                    <th>Покриття</th>
                    <th className="num">Точок</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {byBuilding.get(b)!.map((f) => (
                    <tr key={f.id}>
                      <td className="mono">{facadeName(f.id)}</td>
                      <td>
                        {OCTANT_LABEL[octant(f.id)] ?? octant(f.id)}{" "}
                        <span className="faint num">{f.azimuth_deg.toFixed(0)}°</span>
                      </td>
                      <td className="num">{(f.u_max - f.u_min).toFixed(1)} м</td>
                      <td className="num">{(f.v_max - f.v_ground).toFixed(1)} м</td>
                      <td className="num" title="RMS розкиду точок навколо площини">
                        ±{(100 * f.plane_rms_m).toFixed(0)} см
                      </td>
                      <td>
                        <CoverageBar coverage={f.coverage} />
                      </td>
                      <td className="num">{data.features_by_facade[f.id] ?? 0}</td>
                      <td style={{ textAlign: "right" }}>
                        <button
                          className="btn sm"
                          disabled={!data.scene}
                          // Always pinned: the id means this wall only in this run.
                          onClick={() =>
                            navigate(`/facade/${project}/${site}/${f.id}?run=${encodeURIComponent(data.run_id)}`)
                          }
                        >
                          <Icon name="camera" size={14} /> Редактор
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        );
      })}

      {data.facades.length === 0 && (
        <div className="empty">
          <div className="big">Фасадів не знайдено</div>
          <p className="muted">
            Стадія <code>semantics</code> пройшла, але жодна стіна не набрала потрібної довжини, висоти
            чи кількості точок. Параметри — у конфігу run-а (<code>facade_min_length</code>,{" "}
            <code>facade_min_height</code>, <code>facade_min_points</code>).
          </p>
        </div>
      )}
    </div>
  );
}
