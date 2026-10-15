import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Link, Route, Routes } from "react-router-dom";

import { Shell } from "./components/Shell";
import { Dashboard } from "./pages/Dashboard";
import { JobPage } from "./pages/JobPage";
import { JobsPage } from "./pages/JobsPage";
import { SitePage } from "./pages/SitePage";
import { ViewerPage } from "./pages/ViewerPage";
import "./styles.css";

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: true, staleTime: 1000 } },
});

function NotFound() {
  return (
    <div className="page">
      <div className="empty">
        <div className="big">Сторінку не знайдено</div>
        <Link to="/">До ділянок</Link>
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter basename="/app">
        <Routes>
          <Route element={<Shell />}>
            <Route index element={<Dashboard />} />
            <Route path="sites/:project/:site" element={<SitePage />} />
            <Route path="viewer/:project/:site" element={<ViewerPage />} />
            <Route path="jobs" element={<JobsPage />} />
            <Route path="jobs/:id" element={<JobPage />} />
            <Route path="*" element={<NotFound />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
);
