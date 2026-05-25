import React, { useEffect, useState } from "react";

const API_BASE = "/frontend-api";

const exampleData = {
  bookingId: "INN02501",
  reservationDate: "2018-04-08",
  adults: 1,
  children: 0,
  weekendNights: 0,
  weekNights: 1,
  meal: "Meal Plan 1",
  parking: "0",
  roomType: "Room_Type 1",
  leadTime: 4,
  marketSegment: "Online",
  repeated: "0",
  previousCanceled: 0,
  previousNotCanceled: 0,
  averagePrice: 95,
  specialRequests: 1,
};

const emptyData = {
  bookingId: "",
  reservationDate: "",
  adults: 1,
  children: 0,
  weekendNights: 0,
  weekNights: 1,
  meal: "Meal Plan 1",
  parking: "0",
  roomType: "Room_Type 1",
  leadTime: 0,
  marketSegment: "Online",
  repeated: "0",
  previousCanceled: 0,
  previousNotCanceled: 0,
  averagePrice: 0,
  specialRequests: 0,
};

const batchExampleBookings = [
  exampleData,
  {
    bookingId: "INN04192",
    reservationDate: "2018-09-14",
    adults: 2,
    children: 0,
    weekendNights: 1,
    weekNights: 2,
    meal: "Meal Plan 1",
    parking: "0",
    roomType: "Room_Type 4",
    leadTime: 118,
    marketSegment: "Online",
    repeated: "0",
    previousCanceled: 0,
    previousNotCanceled: 0,
    averagePrice: 142,
    specialRequests: 0,
  },
  {
    bookingId: "INN01877",
    reservationDate: "2018-07-21",
    adults: 2,
    children: 1,
    weekendNights: 1,
    weekNights: 3,
    meal: "Meal Plan 2",
    parking: "0",
    roomType: "Room_Type 1",
    leadTime: 39,
    marketSegment: "Offline",
    repeated: "0",
    previousCanceled: 0,
    previousNotCanceled: 1,
    averagePrice: 88,
    specialRequests: 1,
  },
  {
    bookingId: "INN06340",
    reservationDate: "2018-11-02",
    adults: 2,
    children: 0,
    weekendNights: 2,
    weekNights: 4,
    meal: "Meal Plan 1",
    parking: "0",
    roomType: "Room_Type 2",
    leadTime: 214,
    marketSegment: "Online",
    repeated: "0",
    previousCanceled: 2,
    previousNotCanceled: 0,
    averagePrice: 156,
    specialRequests: 0,
  },
  {
    bookingId: "INN03016",
    reservationDate: "2018-05-19",
    adults: 1,
    children: 0,
    weekendNights: 0,
    weekNights: 2,
    meal: "Meal Plan 1",
    parking: "0",
    roomType: "Room_Type 1",
    leadTime: 16,
    marketSegment: "Corporate",
    repeated: "1",
    previousCanceled: 0,
    previousNotCanceled: 3,
    averagePrice: 72,
    specialRequests: 2,
  },
];

const csvColumns = [
  "Booking_ID",
  "number of adults",
  "number of children",
  "number of weekend nights",
  "number of week nights",
  "type of meal",
  "car parking space",
  "room type",
  "lead time",
  "market segment type",
  "repeated",
  "P-C",
  "P-not-C",
  "average price",
  "special requests",
  "date of reservation",
];

const inputStyle = {
  width: "100%",
  boxSizing: "border-box",
  border: "1px solid #d7dde8",
  borderRadius: 12,
  padding: "10px 12px",
  fontSize: 15,
  background: "#fff",
  color: "#0f172a",
  outline: "none",
};

const buttonStyle = {
  border: "none",
  borderRadius: 12,
  padding: "11px 16px",
  fontSize: 14,
  fontWeight: 700,
  cursor: "pointer",
};

async function requestJson(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  const contentType = response.headers.get("content-type") || "";
  const body = contentType.includes("application/json") ? await response.json() : null;

  if (!response.ok) {
    const message = body?.detail || `Запрос завершился с ошибкой: ${response.status}`;
    throw new Error(message);
  }

  return body;
}

function parseNumber(value, fallback = 0) {
  const normalized = Number(value);
  return Number.isFinite(normalized) ? normalized : fallback;
}

function normalizeBooking(rawBooking) {
  return {
    bookingId: String(rawBooking.bookingId || ""),
    reservationDate: String(rawBooking.reservationDate || ""),
    adults: parseNumber(rawBooking.adults, 0),
    children: parseNumber(rawBooking.children, 0),
    weekendNights: parseNumber(rawBooking.weekendNights, 0),
    weekNights: parseNumber(rawBooking.weekNights, 0),
    meal: String(rawBooking.meal || "Meal Plan 1"),
    parking: String(rawBooking.parking ?? "0"),
    roomType: String(rawBooking.roomType || "Room_Type 1"),
    leadTime: parseNumber(rawBooking.leadTime, 0),
    marketSegment: String(rawBooking.marketSegment || "Online"),
    repeated: String(rawBooking.repeated ?? "0"),
    previousCanceled: parseNumber(rawBooking.previousCanceled, 0),
    previousNotCanceled: parseNumber(rawBooking.previousNotCanceled, 0),
    averagePrice: parseNumber(rawBooking.averagePrice, 0),
    specialRequests: parseNumber(rawBooking.specialRequests, 0),
  };
}

function bookingToCsvRow(booking) {
  return [
    booking.bookingId,
    booking.adults,
    booking.children,
    booking.weekendNights,
    booking.weekNights,
    booking.meal,
    booking.parking,
    booking.roomType,
    booking.leadTime,
    booking.marketSegment,
    booking.repeated,
    booking.previousCanceled,
    booking.previousNotCanceled,
    booking.averagePrice,
    booking.specialRequests,
    booking.reservationDate,
  ];
}

function buildTemplateCsv() {
  return `${csvColumns.join(",")}\n`;
}

function downloadTextFile(filename, content) {
  const blob = new Blob([content], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function parseCsvLine(line) {
  const values = [];
  let current = "";
  let inQuotes = false;

  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];

    if (char === '"') {
      if (inQuotes && line[index + 1] === '"') {
        current += '"';
        index += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === "," && !inQuotes) {
      values.push(current.trim());
      current = "";
    } else {
      current += char;
    }
  }

  values.push(current.trim());
  return values;
}

function parseCsvBookings(text) {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (lines.length < 2) {
    throw new Error("CSV-файл пустой или не содержит строк с данными.");
  }

  const headers = parseCsvLine(lines[0]);
  const requiredHeaders = new Set(csvColumns);

  for (const header of requiredHeaders) {
    if (!headers.includes(header)) {
      throw new Error(`В CSV отсутствует обязательная колонка: ${header}`);
    }
  }

  return lines.slice(1).map((line) => {
    const values = parseCsvLine(line);
    const row = Object.fromEntries(headers.map((header, index) => [header, values[index] || ""]));

    return normalizeBooking({
      bookingId: row["Booking_ID"],
      reservationDate: row["date of reservation"],
      adults: row["number of adults"],
      children: row["number of children"],
      weekendNights: row["number of weekend nights"],
      weekNights: row["number of week nights"],
      meal: row["type of meal"],
      parking: row["car parking space"],
      roomType: row["room type"],
      leadTime: row["lead time"],
      marketSegment: row["market segment type"],
      repeated: row["repeated"],
      previousCanceled: row["P-C"],
      previousNotCanceled: row["P-not-C"],
      averagePrice: row["average price"],
      specialRequests: row["special requests"],
    });
  });
}

function getRiskMeta(risk) {
  if (risk >= 60) {
    return { color: "#dc2626", background: "#fef2f2", label: "Высокий" };
  }

  if (risk >= 35) {
    return { color: "#d97706", background: "#fffbeb", label: "Средний" };
  }

  return { color: "#059669", background: "#ecfdf5", label: "Низкий" };
}

function Field({ label, hint, children }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <label
        style={{
          fontSize: 15,
          fontWeight: 700,
          color: "#334155",
          lineHeight: 1.25,
          minHeight: 28,
          display: "block",
        }}
      >
        {label}
      </label>
      {children}
      {hint ? <span style={{ fontSize: 13, color: "#64748b" }}>{hint}</span> : null}
    </div>
  );
}

function Section({ title, icon, description, children }) {
  return (
    <section
      style={{
        background: "#fff",
        border: "1px solid #e2e8f0",
        borderRadius: 24,
        padding: 20,
        boxShadow: "0 10px 24px rgba(15, 23, 42, 0.04)",
      }}
    >
      <div style={{ display: "flex", alignItems: "flex-start", gap: 10, marginBottom: 18 }}>
        <div
          style={{
            width: 36,
            height: 36,
            borderRadius: 14,
            background: "#f1f5f9",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 18,
            flex: "0 0 auto",
          }}
        >
          {icon}
        </div>
        <div>
          <h2 style={{ margin: 0, fontSize: 18 }}>{title}</h2>
          {description ? (
            <p style={{ margin: "5px 0 0", color: "#64748b", fontSize: 14, lineHeight: 1.5 }}>
              {description}
            </p>
          ) : null}
        </div>
      </div>
      {children}
    </section>
  );
}

function SectionGrid({ children }) {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
        gap: 16,
      }}
    >
      {children}
    </div>
  );
}

function TextInput({ value, onChange, type = "text", min, placeholder }) {
  return (
    <input
      style={inputStyle}
      type={type}
      min={min}
      placeholder={placeholder}
      value={value}
      onChange={(event) => onChange(event.target.value)}
    />
  );
}

function SelectInput({ value, onChange, options }) {
  return (
    <select style={inputStyle} value={value} onChange={(event) => onChange(event.target.value)}>
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  );
}

function TabButton({ active, children, onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        ...buttonStyle,
        background: active ? "#0f172a" : "#f1f5f9",
        color: active ? "#fff" : "#334155",
        border: active ? "1px solid #0f172a" : "1px solid #e2e8f0",
      }}
    >
      {children}
    </button>
  );
}

function StatusPill({ children, tone = "success" }) {
  const styles =
    tone === "error"
      ? {
          color: "#b91c1c",
          background: "#fef2f2",
          border: "1px solid #fecaca",
          dot: "#ef4444",
        }
      : tone === "warning"
        ? {
            color: "#b45309",
            background: "#fffbeb",
            border: "1px solid #fde68a",
            dot: "#f59e0b",
          }
        : {
            color: "#059669",
            background: "#ecfdf5",
            border: "1px solid #bbf7d0",
            dot: "#22c55e",
          };

  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 7,
        color: styles.color,
        background: styles.background,
        border: styles.border,
        borderRadius: 999,
        padding: "7px 11px",
        fontSize: 13,
        fontWeight: 800,
      }}
    >
      <span
        style={{
          width: 8,
          height: 8,
          borderRadius: 999,
          background: styles.dot,
          display: "inline-block",
        }}
      />
      {children}
    </span>
  );
}

function SummaryCard({ label, value, hint }) {
  return (
    <div
      style={{
        background: "#f8fafc",
        border: "1px solid #e2e8f0",
        borderRadius: 20,
        padding: 18,
      }}
    >
      <p style={{ margin: 0, color: "#64748b", fontSize: 13, fontWeight: 700 }}>{label}</p>
      
      <p
        style={{
          margin: "8px 0 0",
          color: "#0f172a",
          fontSize: 32,
          fontWeight: 850,
          letterSpacing: -0.7,
        }}
      >
        {value}
      </p>
      {hint ? <p style={{ margin: "6px 0 0", color: "#64748b", fontSize: 13 }}>{hint}</p> : null}
    </div>
  );
}

function RiskBadge({ risk }) {
  const meta = getRiskMeta(risk);

  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
        borderRadius: 999,
        padding: "6px 10px",
        background: meta.background,
        color: meta.color,
        fontWeight: 800,
        fontSize: 12,
      }}
    >
      {meta.label} · {risk}%
    </span>
  );
}

function SingleScoringPanel() {
  const [form, setForm] = useState(exampleData);
  const [result, setResult] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState("");

  const update = (key, value) => {
    setForm((previous) => ({ ...previous, [key]: value }));
    setResult(null);
    setError("");
  };

  const calculateRisk = async () => {
    setIsSubmitting(true);
    setError("");

    try {
      const prediction = await requestJson("/predict", {
        method: "POST",
        body: JSON.stringify(normalizeBooking(form)),
      });
      setResult(prediction);
    } catch (requestError) {
      setError(requestError.message);
      setResult(null);
    } finally {
      setIsSubmitting(false);
    }
  };

  const risk = result?.risk ?? null;
  const riskMeta = risk === null ? null : getRiskMeta(risk);
  const riskLabel = risk === null ? "Не рассчитан" : `${riskMeta.label} риск отмены`;
  const riskColor = riskMeta?.color || "#64748b";
  const recommendation =
    risk === null
      ? "Заполните данные бронирования и запустите скоринг, чтобы получить прогноз."
      : risk >= 60
        ? "Рекомендуемое действие: заранее подтвердить бронирование или применить дополнительные меры контроля риска."
        : risk >= 35
          ? "Рекомендуемое действие: отслеживать это бронирование и при необходимости отправить мягкое напоминание перед заездом."
          : "Рекомендуемое действие: оставить стандартную обработку бронирования.";

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "minmax(0, 1fr) minmax(300px, 360px)",
        gap: 24,
        alignItems: "start",
      }}
    >
      <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
        <Section
          title="Основная информация о бронировании"
          icon="📋"
          description="Укажите идентификатор бронирования и дату резервирования."
        >
          <SectionGrid>
            <Field label="ID бронирования" hint="Необязательно. Если оставить пустым, можно сгенерировать автоматически.">
              <TextInput
                value={form.bookingId}
                placeholder="INN02501"
                onChange={(value) => update("bookingId", value)}
              />
            </Field>
            <Field label="Дата бронирования">
              <TextInput
                type="date"
                value={form.reservationDate}
                onChange={(value) => update("reservationDate", value)}
              />
            </Field>
          </SectionGrid>
        </Section>

        <Section title="Гости" icon="👥" description="Состав гостей для этого бронирования.">
          <SectionGrid>
            <Field label="Количество взрослых">
              <TextInput
                type="number"
                min="0"
                value={form.adults}
                onChange={(value) => update("adults", value)}
              />
            </Field>
            <Field label="Количество детей">
              <TextInput
                type="number"
                min="0"
                value={form.children}
                onChange={(value) => update("children", value)}
              />
            </Field>
          </SectionGrid>
        </Section>

        <Section title="Детали проживания" icon="🛏️" description="Тип номера, питание, длительность проживания и цена.">
          <SectionGrid>
            <Field label="Ночей в выходные">
              <TextInput
                type="number"
                min="0"
                value={form.weekendNights}
                onChange={(value) => update("weekendNights", value)}
              />
            </Field>
            <Field label="Ночей в будни">
              <TextInput
                type="number"
                min="0"
                value={form.weekNights}
                onChange={(value) => update("weekNights", value)}
              />
            </Field>
            <Field label="Тип номера">
              <SelectInput
                value={form.roomType}
                onChange={(value) => update("roomType", value)}
                options={[
                  { value: "Room_Type 1", label: "Номер типа 1" },
                  { value: "Room_Type 2", label: "Номер типа 2" },
                  { value: "Room_Type 3", label: "Номер типа 3" },
                  { value: "Room_Type 4", label: "Номер типа 4" },
                  { value: "Room_Type 5", label: "Номер типа 5" },
                  { value: "Room_Type 6", label: "Номер типа 6" },
                  { value: "Room_Type 7", label: "Номер типа 7" },
                ]}
              />
            </Field>
            <Field label="Тип питания">
              <SelectInput
                value={form.meal}
                onChange={(value) => update("meal", value)}
                options={[
                  { value: "Meal Plan 1", label: "План питания 1" },
                  { value: "Meal Plan 2", label: "План питания 2" },
                  { value: "Meal Plan 3", label: "План питания 3" },
                  { value: "Not Selected", label: "Не выбрано" },
                ]}
              />
            </Field>
            <Field label="Средняя цена">
              <TextInput
                type="number"
                min="0"
                value={form.averagePrice}
                onChange={(value) => update("averagePrice", value)}
              />
            </Field>
          </SectionGrid>
        </Section>

        <Section
          title="История клиента и предпочтения"
          icon="🧾"
          description="Предыдущее поведение клиента и дополнительные параметры бронирования."
        >
          <SectionGrid>
            <Field label="Повторный гость">
              <SelectInput
                value={form.repeated}
                onChange={(value) => update("repeated", value)}
                options={[
                  { value: "0", label: "Нет" },
                  { value: "1", label: "Да" },
                ]}
              />
            </Field>
            <Field label="Парковочное место">
              <SelectInput
                value={form.parking}
                onChange={(value) => update("parking", value)}
                options={[
                  { value: "0", label: "Нет" },
                  { value: "1", label: "Да" },
                ]}
              />
            </Field>
            <Field label="Отмены ранее (P-C)">
              <TextInput
                type="number"
                min="0"
                value={form.previousCanceled}
                onChange={(value) => update("previousCanceled", value)}
              />
            </Field>
            <Field label="Неотменённые ранее (P-not-C)">
              <TextInput
                type="number"
                min="0"
                value={form.previousNotCanceled}
                onChange={(value) => update("previousNotCanceled", value)}
              />
            </Field>
            <Field label="Особые запросы">
              <TextInput
                type="number"
                min="0"
                value={form.specialRequests}
                onChange={(value) => update("specialRequests", value)}
              />
            </Field>
          </SectionGrid>
        </Section>

        <Section
          title="Канал бронирования и сроки"
          icon="📅"
          description="Срок до заезда и рыночный сегмент влияют на вероятность отмены."
        >
          <SectionGrid>
            <Field label="Срок до заезда" hint="Количество дней между бронированием и заездом.">
              <TextInput
                type="number"
                min="0"
                value={form.leadTime}
                onChange={(value) => update("leadTime", value)}
              />
            </Field>
            <Field label="Рыночный сегмент">
              <SelectInput
                value={form.marketSegment}
                onChange={(value) => update("marketSegment", value)}
                options={[
                  { value: "Online", label: "Онлайн" },
                  { value: "Offline", label: "Офлайн" },
                  { value: "Corporate", label: "Корпоративный" },
                  { value: "Complementary", label: "Комплиментарный" },
                  { value: "Aviation", label: "Авиация" },
                ]}
              />
            </Field>
          </SectionGrid>
        </Section>
      </div>

      <aside
        style={{
          position: "sticky",
          top: 104,
          background: "#fff",
          border: "1px solid #e2e8f0",
          borderRadius: 28,
          padding: 22,
          boxShadow: "0 10px 24px rgba(15, 23, 42, 0.05)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 20 }}>
          <div
            style={{
              width: 40,
              height: 40,
              borderRadius: 14,
              background: "#f1f5f9",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            AI
          </div>
          <div>
            <h2 style={{ margin: 0, fontSize: 20 }}>Результат скоринга</h2>
            <p style={{ margin: "3px 0 0", fontSize: 13, color: "#64748b" }}>
              Прогноз для одного бронирования
            </p>
          </div>
        </div>

        <div style={{ borderRadius: 22, background: "#f8fafc", padding: 22, textAlign: "center" }}>
          <p style={{ margin: 0, color: "#64748b", fontSize: 15 }}>Вероятность отмены</p>
          <p style={{ margin: "8px 0", fontSize: 54, fontWeight: 800, letterSpacing: -1.5 }}>
            {risk === null ? "—" : `${risk}%`}
          </p>
          <p style={{ margin: 0, color: riskColor, fontSize: 15, fontWeight: 800 }}>
            {riskLabel}
          </p>
        </div>

        <div
          style={{
            marginTop: 16,
            height: 12,
            borderRadius: 999,
            overflow: "hidden",
            background: "#e2e8f0",
          }}
        >
          <div
            style={{
              height: "100%",
              width: risk === null ? "0%" : `${risk}%`,
              background: riskColor === "#64748b" ? "#0f172a" : riskColor,
              transition: "width 0.3s ease",
            }}
          />
        </div>

        <div
          style={{
            marginTop: 18,
            border: "1px solid #e2e8f0",
            borderRadius: 18,
            padding: 16,
            color: "#475569",
            fontSize: 15,
            lineHeight: 1.5,
          }}
        >
          <strong style={{ color: riskColor }}>{risk !== null && risk >= 60 ? "Высокий риск: " : "Инфо: "}</strong>
          {recommendation}
        </div>

        {error ? (
          <div
            style={{
              marginTop: 14,
              padding: 14,
              borderRadius: 16,
              background: "#fef2f2",
              color: "#b91c1c",
              fontSize: 14,
              border: "1px solid #fecaca",
            }}
          >
            {error}
          </div>
        ) : null}

        <div style={{ marginTop: 18, display: "flex", gap: 10, flexWrap: "wrap" }}>
          <button
            type="button"
            style={{
              ...buttonStyle,
              background: "#fff",
              color: "#0f172a",
              border: "1px solid #cbd5e1",
              flex: 1,
            }}
            onClick={() => {
              setForm(exampleData);
              setResult(null);
              setError("");
            }}
          >
            Заполнить примером
          </button>
          <button
            type="button"
            style={{ ...buttonStyle, background: "#e2e8f0", color: "#0f172a", flex: 1 }}
            onClick={() => {
              setForm(emptyData);
              setResult(null);
              setError("");
            }}
          >
            Очистить
          </button>
          <button
            type="button"
            style={{ ...buttonStyle, background: "#0f172a", color: "#fff", width: "100%" }}
            onClick={calculateRisk}
            disabled={isSubmitting}
          >
            {isSubmitting ? "Расчёт..." : "Рассчитать риск"}
          </button>
        </div>
      </aside>
    </div>
  );
}

function BatchScoringPanel() {
  const [fileName, setFileName] = useState("");
  const [bookings, setBookings] = useState([]);
  const [rows, setRows] = useState([]);
  const [summary, setSummary] = useState({ total: 0, highRiskCount: 0, averageProbability: 0 });
  const [hasRun, setHasRun] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState("");

  const topRiskyRows = [...rows].sort((a, b) => b.risk - a.risk).slice(0, 5);

  const loadExample = () => {
    setFileName("hotel_bookings_example.csv");
    setBookings(batchExampleBookings);
    setRows([]);
    setSummary({ total: 0, highRiskCount: 0, averageProbability: 0 });
    setHasRun(false);
    setError("");
  };

  const runBatchScoring = async () => {
    setIsSubmitting(true);
    setError("");

    try {
      const sourceBookings = bookings.length > 0 ? bookings : batchExampleBookings;

      if (bookings.length === 0) {
        setBookings(batchExampleBookings);
        setFileName("hotel_bookings_example.csv");
      }

      const response = await requestJson("/predict/batch", {
        method: "POST",
        body: JSON.stringify({
          riskShare: 0.3,
          bookings: sourceBookings.map(normalizeBooking),
        }),
      });

      setRows(response.predictions || []);
      setSummary(
        response.summary || {
          total: response.predictions?.length || 0,
          highRiskCount: 0,
          averageProbability: 0,
        }
      );
      setHasRun(true);
    } catch (requestError) {
      setError(requestError.message);
      setRows([]);
      setSummary({ total: 0, highRiskCount: 0, averageProbability: 0 });
      setHasRun(false);
    } finally {
      setIsSubmitting(false);
    }
  };

  const downloadTemplate = () => {
    downloadTextFile("booking_scoring_template.csv", buildTemplateCsv());
  };

  const downloadReport = () => {
    const header =
      "Booking_ID,reservation_date,market_segment,room_type,lead_time,average_price,cancellation_probability,risk_group\n";
    const body = rows
      .map(
        (row) =>
          `${row.bookingId},${row.reservationDate},${row.marketSegment},${row.roomType},${row.leadTime},${row.averagePrice},${row.risk},${row.riskSegment}`
      )
      .join("\n");
    downloadTextFile("batch_scoring_report.csv", header + body);
  };

  const reportReady = hasRun && rows.length > 0;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      <Section
        title="Загрузка CSV с бронированиями"
        icon="📁"
        description="Загрузите датасет с той же схемой, которую использует модель."
      >
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "minmax(0, 1fr) 260px",
            gap: 18,
            alignItems: "stretch",
          }}
        >
          <label
            style={{
              border: "2px dashed #cbd5e1",
              borderRadius: 22,
              background: "#f8fafc",
              padding: 24,
              display: "flex",
              flexDirection: "column",
              justifyContent: "center",
              gap: 8,
              cursor: "pointer",
            }}
          >
            <input
              type="file"
              accept=".csv"
              style={{ display: "none" }}
              onChange={async (event) => {
                const file = event.target.files && event.target.files[0];
                if (!file) {
                  return;
                }

                try {
                  const text = await file.text();
                  const parsedBookings = parseCsvBookings(text);
                  setFileName(file.name);
                  setBookings(parsedBookings);
                  setRows([]);
                  setSummary({ total: 0, highRiskCount: 0, averageProbability: 0 });
                  setHasRun(false);
                  setError("");
                } catch (fileError) {
                  setError(fileError.message);
                  setFileName(file.name);
                  setBookings([]);
                  setRows([]);
                  setSummary({ total: 0, highRiskCount: 0, averageProbability: 0 });
                  setHasRun(false);
                }
              }}
            />
            <strong style={{ fontSize: 18 }}>Перетащите CSV сюда или нажмите для загрузки</strong>
            <span style={{ color: "#64748b", fontSize: 14 }}>
              Ожидаемые колонки: детали бронирования, данные о гостях, параметры проживания и история клиента.
            </span>
            <span
              style={{
                color: fileName ? "#64748b" : "#94a3b8",
                fontSize: 14,
                fontWeight: 700,
              }}
            >
              {fileName || "Файл не выбран"}
            </span>
            <span style={{ color: "#64748b", fontSize: 13 }}>
              Загружено строк: {bookings.length}
            </span>
          </label>

          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <button
              type="button"
              style={{ ...buttonStyle, background: "#0f172a", color: "#fff" }}
              onClick={loadExample}
            >
              Загрузить пример данных
            </button>
            <button
              type="button"
              style={{
                ...buttonStyle,
                background: "#fff",
                color: "#0f172a",
                border: "1px solid #cbd5e1",
              }}
              onClick={downloadTemplate}
            >
              Скачать пустой шаблон
            </button>
            <p style={{ margin: 0, color: "#64748b", fontSize: 13, lineHeight: 1.45 }}>
              Загрузите свой файл или пример данных, чтобы быстро посмотреть сценарий работы.
            </p>
          </div>
        </div>
      </Section>

      <Section
        title="Запуск батч-скоринга"
        icon="⚙️"
        description="Рассчитайте вероятность отмены для каждой загруженной строки."
      >
        <div
          style={{
            display: "flex",
            justifyContent: "flex-end",
            alignItems: "flex-start",
            gap: 12,
            flexWrap: "wrap",
            marginTop: -52,
          }}
        >
          <button
            type="button"
            style={{
              ...buttonStyle,
              background: bookings.length > 0 ? "#0f172a" : "#334155",
              color: "#fff",
              width: 260,
            }}
            onClick={runBatchScoring}
            disabled={isSubmitting}
          >
            {isSubmitting
              ? "Запуск..."
              : bookings.length > 0
                ? "Запустить батч-скоринг"
                : "Запустить на примере"}
          </button>
        </div>

        {error ? (
          <div
            style={{
              marginTop: 16,
              padding: 14,
              borderRadius: 16,
              background: "#fef2f2",
              color: "#b91c1c",
              fontSize: 13,
              border: "1px solid #fecaca",
            }}
          >
            {error}
          </div>
        ) : null}
      </Section>

      <Section title="Сводка по батч-скорингу" icon="📊" description="Основные метрики по результатам запуска скоринга.">
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))",
            gap: 14,
          }}
        >
          <SummaryCard label="Всего предсказаний" value={hasRun ? summary.total : "—"} hint="Строк обработано из CSV" />
          <SummaryCard label="Высокий риск" value={hasRun ? summary.highRiskCount : "—"} hint="Бронирования с риском >= 60%" />
          <SummaryCard
            label="Средняя вероятность"
            value={hasRun ? `${summary.averageProbability}%` : "—"}
            hint="Средняя вероятность отмены"
          />
        </div>
      </Section>

      <Section title="Бронирования с наибольшим риском" icon="🚩" description="Бронирования, которые стоит проверить в первую очередь.">
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
            <thead>
              <tr style={{ textAlign: "left", color: "#64748b", borderBottom: "1px solid #e2e8f0" }}>
                <th style={{ padding: "12px 10px" }}>ID бронирования</th>
                <th style={{ padding: "12px 10px" }}>Дата бронирования</th>
                <th style={{ padding: "12px 10px" }}>Сегмент</th>
                <th style={{ padding: "12px 10px" }}>Тип номера</th>
                <th style={{ padding: "12px 10px" }}>Срок до заезда</th>
                <th style={{ padding: "12px 10px" }}>Средняя цена</th>
                <th style={{ padding: "12px 10px" }}>Риск</th>
              </tr>
            </thead>
            <tbody>
              {hasRun && topRiskyRows.length > 0 ? (
                topRiskyRows.map((row) => (
                  <tr key={`${row.bookingId}-${row.reservationDate}`} style={{ borderBottom: "1px solid #f1f5f9" }}>
                    <td style={{ padding: "14px 10px", fontWeight: 800 }}>{row.bookingId}</td>
                    <td style={{ padding: "14px 10px", color: "#475569" }}>{row.reservationDate}</td>
                    <td style={{ padding: "14px 10px", color: "#475569" }}>{row.marketSegment}</td>
                    <td style={{ padding: "14px 10px", color: "#475569" }}>{row.roomType}</td>
                    <td style={{ padding: "14px 10px", color: "#475569" }}>{row.leadTime} дн.</td>
                    <td style={{ padding: "14px 10px", color: "#475569" }}>{row.averagePrice}</td>
                    <td style={{ padding: "14px 10px" }}>
                      <RiskBadge risk={row.risk} />
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="7" style={{ padding: 24, textAlign: "center", color: "#94a3b8" }}>
                    Запустите батч-скоринг, чтобы увидеть самые рискованные бронирования.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Section>

      <Section
        title="Экспорт результатов предсказания"
        icon="⬇️"
        description="Отчёт включает ID бронирований, вероятности и группы риска."
      >
        <div
          style={{
            display: "flex",
            justifyContent: "flex-end",
            alignItems: "flex-start",
            gap: 18,
            flexWrap: "wrap",
            marginTop: -52,
          }}
        >
          <button
            type="button"
            disabled={!reportReady}
            style={{
              ...buttonStyle,
              background: reportReady ? "#0f172a" : "#e2e8f0",
              color: reportReady ? "#fff" : "#94a3b8",
              width: 260,
              cursor: reportReady ? "pointer" : "not-allowed",
            }}
            onClick={downloadReport}
          >
            Скачать отчёт
          </button>
        </div>
      </Section>
    </div>
  );
}

function HomePage({ onNavigateToPredict, health }) {
  const features = [
    {
      title: "Одиночный скоринг",
      description:
        "Оценка вероятности отмены для одного бронирования с помощью структурированной формы.",
      icon: "🎯",
    },
    {
      title: "Батч-скоринг",
      description: "Загрузите CSV и обработайте несколько гостиничных бронирований за один запуск.",
      icon: "📁",
    },
    {
      title: "Приоритизация риска",
      description: "Сначала просматривайте бронирования с высоким риском и быстрее принимайте операционные решения.",
      icon: "📊",
    },
  ];

  const statusTone = health.error ? "error" : health.modelLoaded ? "success" : "warning";
  const statusLabel = health.error
    ? "Бэкенд недоступен"
    : health.modelLoaded
      ? "Сервис доступен"
      : "Модель ещё не загружена";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      <section
        style={{
          background: "linear-gradient(135deg, #0f172a 0%, #1e293b 100%)",
          color: "#fff",
          borderRadius: 32,
          padding: 36,
          position: "relative",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            position: "absolute",
            right: -60,
            top: -60,
            width: 220,
            height: 220,
            borderRadius: "50%",
            background: "rgba(255,255,255,0.05)",
          }}
        />

        <div style={{ position: "relative", zIndex: 1, maxWidth: 760 }}>
          <StatusPill tone={statusTone}>{statusLabel}</StatusPill>
          <h1 style={{ margin: "18px 0 0", fontSize: 52, lineHeight: 1, letterSpacing: -2 }}>
            Прогнозирование отмены гостиничных бронирований
          </h1>
          <p style={{ margin: "18px 0 0", color: "#cbd5e1", fontSize: 17, lineHeight: 1.7 }}>
            Сервис на базе ML для прогнозирования отмены гостиничных бронирований. Используйте его, чтобы выявлять рискованные брони, расставлять приоритеты в работе и снижать потенциальные потери выручки.
          </p>

          <div style={{ display: "flex", gap: 12, marginTop: 28, flexWrap: "wrap" }}>
            <button
              type="button"
              onClick={onNavigateToPredict}
              style={{
                ...buttonStyle,
                background: "#fff",
                color: "#0f172a",
                padding: "14px 22px",
                fontSize: 15,
              }}
            >
              Перейти к прогнозу
            </button>
          </div>
        </div>
      </section>

      <section
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
          gap: 18,
        }}
      >
        {features.map((feature) => (
          <div
            key={feature.title}
            style={{
              background: "#fff",
              border: "1px solid #e2e8f0",
              borderRadius: 24,
              padding: 24,
              boxShadow: "0 10px 24px rgba(15, 23, 42, 0.04)",
            }}
          >
            <div
              style={{
                width: 48,
                height: 48,
                borderRadius: 18,
                background: "#f1f5f9",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 24,
                fontWeight: 800,
                marginBottom: 18,
              }}
            >
              {feature.icon}
            </div>
            <h3 style={{ margin: 0, fontSize: 22 }}>{feature.title}</h3>
            <p style={{ margin: "10px 0 0", color: "#64748b", fontSize: 15, lineHeight: 1.65 }}>
              {feature.description}
            </p>
          </div>
        ))}
      </section>

      <section
        style={{
          background: "#fff",
          border: "1px solid #e2e8f0",
          borderRadius: 28,
          padding: 28,
          boxShadow: "0 10px 24px rgba(15, 23, 42, 0.04)",
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: 20,
          }}
        >
          <div>
            <p style={{ margin: 0, color: "#64748b", fontWeight: 700, fontSize: 14 }}>
              Статус системы
            </p>
            <h2 style={{ margin: "8px 0 0", fontSize: 30 }}>
              {health.error ? "Проблема с подключением к бэкенду" : "Готово к предсказаниям"}
            </h2>
            <p
              style={{
                margin: "10px 0 0",
                color: "#64748b",
                fontSize: 15,
                lineHeight: 1.6,
                maxWidth: 620,
              }}
            >
              {health.error
                ? health.error
                : `Сервис доступен, статус модели: ${health.modelLoaded ? "загружена" : "ожидается"}, API предсказаний доступен.`}
            </p>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 12, minWidth: 280 }}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                background: "#f8fafc",
                border: "1px solid #e2e8f0",
                borderRadius: 18,
                padding: "14px 16px",
              }}
            >
              <span style={{ fontWeight: 700 }}>Сервис</span>
              <span style={{ color: health.error ? "#dc2626" : "#059669", fontWeight: 800 }}>
                {health.error ? "Недоступен" : "Доступен"}
              </span>
            </div>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                background: "#f8fafc",
                border: "1px solid #e2e8f0",
                borderRadius: 18,
                padding: "14px 16px",
              }}
            >
              <span style={{ fontWeight: 700 }}>Модель</span>
              <span style={{ color: health.modelLoaded ? "#059669" : "#d97706", fontWeight: 800 }}>
                {health.modelLoaded ? health.modelName || "Загружена" : "Не загружена"}
              </span>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

function PredictPage() {
  const [activeTab, setActiveTab] = useState("single");

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      <header
        style={{
          background: "#fff",
          borderRadius: 28,
          padding: 24,
          boxShadow: "0 10px 24px rgba(15, 23, 42, 0.05)",
          display: "flex",
          flexDirection: "column",
          alignItems: "flex-start",
          gap: 18,
        }}
      >
        <div>
          <h1 style={{ margin: 0, fontSize: 36, letterSpacing: -0.8 }}>Прогноз</h1>
          <p
            style={{
              margin: "10px 0 0",
              maxWidth: 720,
              fontSize: 16,
              lineHeight: 1.6,
              color: "#64748b",
            }}
          >
            Выберите ручной ввод для одного бронирования или загрузите CSV для массового расчёта.
          </p>
        </div>

        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          <TabButton active={activeTab === "single"} onClick={() => setActiveTab("single")}>
            Одиночный скоринг
          </TabButton>
          <TabButton active={activeTab === "batch"} onClick={() => setActiveTab("batch")}>
            Батч-скоринг
          </TabButton>
        </div>
      </header>

      {activeTab === "single" ? <SingleScoringPanel /> : <BatchScoringPanel />}
    </div>
  );
}

export default function App() {
  const [page, setPage] = useState("home");
  const [health, setHealth] = useState({
    status: "",
    modelLoaded: false,
    modelName: "",
    error: "",
  });

  useEffect(() => {
    let isMounted = true;

    requestJson("/health")
      .then((response) => {
        if (!isMounted) {
          return;
        }

        setHealth({
          status: response.status || "",
          modelLoaded: Boolean(response.modelLoaded),
          modelName: response.modelName || "",
          error: "",
        });
      })
      .catch((requestError) => {
        if (!isMounted) {
          return;
        }

        setHealth({
          status: "",
          modelLoaded: false,
          modelName: "",
          error: requestError.message,
        });
      });

    return () => {
      isMounted = false;
    };
  }, []);

  return (
    <main
      style={{
        minHeight: "100vh",
        background: "#f8fafc",
        color: "#0f172a",
        padding: 24,
        fontFamily: "Inter, Arial, sans-serif",
      }}
    >
      <div
        style={{
          maxWidth: 1240,
          margin: "0 auto",
          display: "flex",
          flexDirection: "column",
          gap: 24,
        }}
      >
        <nav
          style={{
            background: "rgba(255,255,255,0.92)",
            backdropFilter: "blur(12px)",
            border: "1px solid #e2e8f0",
            borderRadius: 24,
            padding: "14px 18px",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            position: "sticky",
            top: 18,
            zIndex: 20,
            gap: 18,
            flexWrap: "wrap",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 20, flexWrap: "wrap" }}>
            <div style={{ display: "flex", gap: 8 }}>
              <TabButton active={page === "home"} onClick={() => setPage("home")}>
                Главная
              </TabButton>
              <TabButton active={page === "predict"} onClick={() => setPage("predict")}>
                Прогноз
              </TabButton>
            </div>
          </div>

          <StatusPill tone={health.error ? "error" : health.modelLoaded ? "success" : "warning"}>
            {health.error ? "Бэкенд недоступен" : health.modelLoaded ? "Сервис доступен" : "Модель загружается"}
          </StatusPill>
        </nav>

        {page === "home" ? (
          <HomePage onNavigateToPredict={() => setPage("predict")} health={health} />
        ) : (
          <PredictPage />
        )}
      </div>
    </main>
  );
}
