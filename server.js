import "dotenv/config";
import express from "express";
import cors from "cors";
import multer from "multer";
import OpenAI from "openai";
import fs from "fs";
import path from "path";
import os from "os";
import Tesseract from "tesseract.js";
import clinicalHistoryRouter from "./clinical-history/routes/clinicalHistory.routes.js";

// ✅ Postgres pool (ya lo tienes en db/pool.js)
import { pool } from "./db/pool.js";

// =======================================================
// pdf-parse — SIMPLE y COMPATIBLE (pdf-parse@1.1.1)
// =======================================================
import { createRequire } from "module";
const require = createRequire(import.meta.url);
const parsePdf = require("pdf-parse"); // ✅ con 1.1.1 esto es FUNCIÓN

const app = express();

// =======================================================
// MIDDLEWARE
// =======================================================
app.use(cors());
app.use(express.json({ limit: "25mb" }));

// =======================================================
// HEALTH CHECK
// =======================================================
app.get("/health", (req, res) => {
  res.json({ ok: true });
});

// ✅ Debug DB (para confirmar en Render si está cogiendo DATABASE_URL)
app.get("/debug/db", async (req, res) => {
  const hasDb = !!process.env.DATABASE_URL;
  const pgSslMode = process.env.PGSSLMODE || null;

  if (!hasDb) {
    return res.json({ hasDb: false, pgSslMode, dbPing: false });
  }

  try {
    const r = await pool.query("select now() as now");
    return res.json({
      hasDb: true,
      pgSslMode,
      dbPing: true,
      now: r.rows?.[0]?.now || null,
    });
  } catch (e) {
    return res.status(500).json({
      hasDb: true,
      pgSslMode,
      dbPing: false,
      error: e?.message || "db_ping_error",
    });
  }
});

// =======================================================
// CLINICAL HISTORY ROUTES (PERSISTENCIA)
// - Esto es lo que te da “permanencia total” de seguimientos
// - Rutas típicas del router:
//   POST /sync/patients/:patientId/history/get-or-create
//   POST /sync/patients/:patientId/history/entries
//   GET  /sync/patients/:patientId/history/entries?limit=50
// =======================================================
app.use(clinicalHistoryRouter);

// =======================================================
// MULTER (uploads: PDF, imágenes, audio)
// =======================================================
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 25 * 1024 * 1024 }, // 25 MB
});

// =======================================================
// OPENAI CLIENT
// =======================================================
const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// =======================================================
// 0) EXTRACT — PDF / IMÁGENES → TEXTO
// =======================================================
app.post("/extract", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "No file provided" });

    const mime = (req.file.mimetype || "").toLowerCase();
    const name = (req.file.originalname || "").toLowerCase();

    const isPdf = mime === "application/pdf" || name.endsWith(".pdf");
    const isImage = mime.startsWith("image/"); // jpg, png, jpeg, heic, webp, tiff, bmp...

    // ---------- PDF ----------
    if (isPdf) {
      const data = await parsePdf(req.file.buffer);
      const text = (data?.text || "").trim();

      return res.json({
        text,
        method: "pdf-parse",
        scanned: text.length < 30,
      });
    }

    // ---------- IMÁGENES (OCR) ----------
    if (isImage) {
      const result = await Tesseract.recognize(req.file.buffer, "spa+eng", {
        logger: () => {},
      });

      return res.json({
        text: (result?.data?.text || "").trim(),
        method: "tesseract",
      });
    }

    return res.status(415).json({
      error: "Unsupported file type (solo PDF o imágenes)",
      mime,
    });
  } catch (e) {
    res.status(500).json({ error: e?.message || "extract_error" });
  }
});

// =======================================================
// 1) TRANSCRIPCIÓN DE AUDIO (M4A / WAV / MP3)
// =======================================================
app.post("/transcribe", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "No file provided" });

    // ✅ En Render, /tmp existe
    const safeName = req.file.originalname || "audio.m4a";
    const tmpPath = path.join(os.tmpdir(), safeName);

    fs.writeFileSync(tmpPath, req.file.buffer);

    const transcription = await client.audio.transcriptions.create({
      file: fs.createReadStream(tmpPath),
      model: process.env.OPENAI_AUDIO_MODEL || "gpt-4o-mini-transcribe",
    });

    try {
      fs.unlinkSync(tmpPath);
    } catch {}

    res.json({ text: transcription?.text || "" });
  } catch (e) {
    res.status(500).json({ error: e?.message || "transcribe_error" });
  }
});

// =======================================================
// 2) REFINAR SOAP + RESUMEN + RP (IA CLÍNICA)
// =======================================================
app.post("/refine", async (req, res) => {
  try {
    const { doctorText = "", attachmentsText = "", transcriptText = "" } = req.body || {};

    const prompt = `
Eres un asistente clínico profesional. NO inventes datos.

Con la información disponible produce:
1) SOAP (S, O, A, P)
2) Resumen ejecutivo
3) RP (Recomendaciones / Plan)

Si falta información escribe "NR".

Devuelve SOLO JSON válido EXACTO:
{
  "soap": { "S":"", "O":"", "A":"", "P":"" },
  "summary": "",
  "rp": ""
}

--- DOCTOR TEXT ---
${doctorText}

--- ADJUNTOS ---
${attachmentsText}

--- AUDIO ---
${transcriptText}
`.trim();

    const r = await client.responses.create({
      model: (process.env.OPENAI_MODEL || "gpt-4.1-mini").trim(),
      input: [
        { role: "system", content: "Devuelve únicamente JSON válido. No texto adicional. No markdown." },
        { role: "user", content: prompt },
      ],
    });

    const out = r.output_text || "{}";

    let parsed;
    try {
      parsed = JSON.parse(out);
    } catch {
      parsed = { error: "invalid_json_from_model", raw: out };
    }

    return res.json(parsed);
  } catch (e) {
    res.status(500).json({ error: e?.message || "refine_error" });
  }
});

// =======================================================
// SERVER (Render-friendly)
// =======================================================
const PORT = Number(process.env.PORT || 8787);
app.listen(PORT, "0.0.0.0", () => {
  console.log(`✅ Mayu backend escuchando en 0.0.0.0:${PORT}`);
});
