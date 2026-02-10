create extension if not exists "pgcrypto";

create table if not exists clinical_histories (
  id uuid primary key default gen_random_uuid(),
  patient_id uuid not null,
  current_document jsonb not null default '{}'::jsonb,
  version int not null default 1,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (patient_id)
);

create table if not exists clinical_history_entries (
  id uuid primary key default gen_random_uuid(),
  history_id uuid not null references clinical_histories(id) on delete cascade,
  patient_id uuid not null,
  appointment_id uuid null,
  author_user_id uuid null,
  source text not null default 'doctor',
  note_raw text not null default '',
  note_structured jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists idx_entries_patient_date
on clinical_history_entries (patient_id, created_at desc);

create index if not exists idx_entries_history_date
on clinical_history_entries (history_id, created_at desc);
