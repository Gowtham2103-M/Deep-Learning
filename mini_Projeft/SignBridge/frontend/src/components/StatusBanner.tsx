import { StatusResponse } from "../api";

export default function StatusBanner({ status, error }: { status: StatusResponse | null; error: string | null }) {
  if (error) return <div className="banner banner-error">{error}</div>;
  if (!status) return <div className="banner">Connecting to the backend...</div>;
  return (
    <>
      {!status.dataset.available && (
        <div className="banner banner-warn">
          <strong>Dataset not found.</strong> {status.dataset.message}
          <br />Speech to ISL and training need the dataset. The rest of the app still works.
        </div>
      )}
      {!status.model.trained && (
        <div className="banner banner-warn">
          <strong>Model not trained yet.</strong> Run the training command (see README, section 7). {status.model.message !== "Model not trained yet. Run the training command." ? status.model.message : ""}
        </div>
      )}
      {status.dataset.available && status.dataset.warnings.map((w) => (
        <div key={w} className="banner banner-info">{w}</div>
      ))}
    </>
  );
}
