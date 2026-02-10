import express from "express";
import { pool } from "../../db/pool.js";

const router = express.Router();

// 1) GET historia viva (si no existe, se crea)
router.get("/sync/patients/:patientId/history", async (req, res) => {
  const { patientId } = req.params;

  try {
    const existing = await pool.query(
      `select id, patient_id, current_document, version
       from clinical_histories
       where patient_id = $1`,
      [patientId]
    );

    if (existing.rows.length === 0) {
      const created = await pool.query(
        `insert into clinical_histories (patient_id, current_document, version)
         values ($1, '{}'::jsonb, 1)
         returning id, patient_id, current_document, version`,
        [patientId]
      );

      const row = created.rows[0];
      return res.json({
        history_id: row.id,
        patient_id: row.patient_id,
        current_document: row.current_document,
        version: row.version,
      });
    }

    const row = existing.rows[0];
    return res.json({
      history_id: row.id,
      patient_id: row.patient_id,
      current_document: row.current_document,
      version: row.version,
    });
  } catch (err) {
    console.error("get history error:", err);
    return res.status(500).json({ error: "server_error" });
  }
});

// 2) POST seguimiento (entry) + update documento vivo con versionado
router.post("/sync/patients/:patientId/history/entries", async (req, res) => {
  const { patientId } = req.params;

  const {
    appointment_id = null,
    author_user_id = null,
    source = "doctor",
    note_raw = "",
    note_structured = {},
    new_current_document = {},
    base_version,
  } = req.body;

  if (base_version === undefined || base_version === null) {
    return res.status(400).json({ error: "base_version_required" });
  }

  const client = await pool.connect();

  try {
    await client.query("BEGIN");

    const historyRes = await client.query(
      `select id, version
       from clinical_histories
       where patient_id = $1
       for update`,
      [patientId]
    );

    if (historyRes.rows.length === 0) {
      await client.query("ROLLBACK");
      return res.status(404).json({ error: "history_not_found" });
    }

    const historyId = historyRes.rows[0].id;
    const currentVersion = historyRes.rows[0].version;

    if (Number(base_version) !== Number(currentVersion)) {
      await client.query("ROLLBACK");
      return res.status(409).json({
        error: "version_conflict",
        server_version: currentVersion,
      });
    }

    const entryRes = await client.query(
      `insert into clinical_history_entries
       (history_id, patient_id, appointment_id, author_user_id, source, note_raw, note_structured)
       values
       ($1, $2, $3, $4, $5, $6, $7::jsonb)
       returning id`,
      [
        historyId,
        patientId,
        appointment_id,
        author_user_id,
        source,
        note_raw,
        JSON.stringify(note_structured),
      ]
    );

    const updatedRes = await client.query(
      `update clinical_histories
       set current_document = $1::jsonb,
           version = version + 1,
           updated_at = now()
       where id = $2
       returning version`,
      [JSON.stringify(new_current_document), historyId]
    );

    await client.query("COMMIT");

    return res.json({
      entry_id: entryRes.rows[0].id,
      history_id: historyId,
      version: updatedRes.rows[0].version,
    });
  } catch (err) {
    await client.query("ROLLBACK");
    console.error("create entry error:", err);
    return res.status(500).json({ error: "server_error" });
  } finally {
    client.release();
  }
});

// 3) GET timeline de seguimientos
router.get("/sync/patients/:patientId/history/entries", async (req, res) => {
  const { patientId } = req.params;
  const limit = Math.min(Number(req.query.limit || 50), 200);

  try {
    const rows = await pool.query(
      `select id, history_id, patient_id, appointment_id, author_user_id, source,
              note_raw, note_structured, created_at
       from clinical_history_entries
       where patient_id = $1
       order by created_at desc
       limit $2`,
      [patientId, limit]
    );

    return res.json({ items: rows.rows });
  } catch (err) {
    console.error("list entries error:", err);
    return res.status(500).json({ error: "server_error" });
  }
});

export default router;
