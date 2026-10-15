import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { viewerUrl } from "../api/client";
import { refreshScenes } from "../api/hooks";
import { Icon } from "../components/Icon";
import { ErrorBox } from "../components/Modal";

/** The classic WebGL viewer (web/appGLSL.js) in a frame, until the WebGL2 renderer lands (Phase 4.2). */
export function ViewerPage() {
  const { project = "", site = "" } = useParams();
  const scene = `${project}/${site}`;
  const [mode, setMode] = useState<"splat" | "dense">("splat");
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    // Runs finished by a worker are new to the server process: re-read the catalog first.
    setReady(false);
    refreshScenes()
      .then((r) => {
        if (!r.scenes.includes(scene)) throw new Error(`Сцени ${scene} ще немає: потрібен успішний run зі splat/cloud.`);
        setReady(true);
      })
      .catch(setError);
  }, [scene]);

  return (
    <div className="page" style={{ maxWidth: "none", gap: 14 }}>
      <div className="page-head">
        <div>
          <div className="crumbs">
            <Link to="/">Ділянки</Link> / <Link to={`/sites/${scene}`}>{scene}</Link> / 3D
          </div>
          <h1>3D-сцена</h1>
        </div>
        <div className="row">
          <div className="segmented">
            <button className={mode === "splat" ? "on" : ""} onClick={() => setMode("splat")}>
              Gaussian Splatting
            </button>
            <button className={mode === "dense" ? "on" : ""} onClick={() => setMode("dense")}>
              Щільна хмара
            </button>
          </div>
          <a className="btn" href={viewerUrl(scene, mode)} target="_blank" rel="noreferrer">
            <Icon name="external" /> Нова вкладка
          </a>
        </div>
      </div>
      <ErrorBox error={error} />
      {ready && <iframe key={mode} className="viewer-frame" title={`3D ${scene}`} src={viewerUrl(scene, mode)} />}
    </div>
  );
}
