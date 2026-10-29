import React, { useEffect, useState, useRef, useCallback } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Box, Cylinder, Text } from '@react-three/drei';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const WS_URL = process.env.REACT_APP_WS_URL || "ws://localhost:8000/ws";
const API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:8000";

// The agent's reasoning arrives over the WebSocket as agent_decision messages
// produced by a real model. There is no canned-message dictionary any more.

// How the agent's verdict for a site maps to what you see. The scene never
// decides for itself whether a reading is bad -- it renders the agent's
// conclusion, or says plainly that there isn't one yet.
const AGENT_STATUS = {
  escalated:   { colour: '#aa1111', dot: '#ff4444', label: 'agent escalated' },
  actioned:    { colour: '#8a5a1a', dot: '#ffaa33', label: 'agent acted' },
  assessed_ok: { colour: '#1a6faf', dot: '#00cc66', label: 'agent reviewed, no action' },
  unassessed:  { colour: '#2a3f55', dot: '#556677', label: 'not yet assessed' },
};

const agentOf = (sc, code) => sc.agent?.[code] || { status: 'unassessed', note: null };
const statusOf = (sc, code) => AGENT_STATUS[agentOf(sc, code).status] || AGENT_STATUS.unassessed;

// Chain health reflects what the AGENT concluded, plus facts it cannot dispute
// (stock levels, delayed shipments). No threshold re-derivation.
function calcHealth(sc) {
  let s = 100;
  Object.values(sc.agent || {}).forEach(a => {
    if (a.status === 'escalated') s -= a.severity === 'critical' ? 25 : 15;
    else if (a.status === 'actioned') s -= 5;
  });
  if (sc.delayedShipments > 0) s -= Math.min(15, sc.delayedShipments * 5);
  if (sc.wh1Inventory < 220) s -= 10;
  return Math.max(0, Math.min(100, s));
}

function Building({ position, color, label, width = 2.5, height = 2 }) {
  return (
    <group position={position}>
      <Box args={[width, height, 2]}>
        <meshStandardMaterial color={color} />
      </Box>
      <Box args={[width + 0.2, 0.1, 2.2]} position={[0, height / 2, 0]}>
        <meshStandardMaterial color="#222" />
      </Box>
      <Text position={[0, height / 2 + 0.45, 0]} fontSize={0.27} color="white" anchorX="center">
        {label}
      </Text>
    </group>
  );
}

function RobotArm({ position, active, loading }) {
  const armRef = useRef();
  useFrame(({ clock }) => {
    if (!armRef.current) return;
    if (loading) armRef.current.rotation.z = Math.sin(clock.getElapsedTime() * 3) * 0.9;
    else if (active) armRef.current.rotation.z = Math.sin(clock.getElapsedTime() * 1.5) * 0.35;
    else armRef.current.rotation.z = 0;
  });
  return (
    <group position={position}>
      <Box args={[0.35, 0.25, 0.35]}>
        <meshStandardMaterial color="#555" />
      </Box>
      <group ref={armRef} position={[0, 0.12, 0]}>
        <Cylinder args={[0.07, 0.07, 1.3, 8]} position={[0, 0.65, 0]}>
          <meshStandardMaterial color={loading ? '#ffaa00' : active ? '#00dd77' : '#cc2222'} />
        </Cylinder>
        {loading && (
          <Box args={[0.3, 0.3, 0.3]} position={[0, 1.35, 0]}>
            <meshStandardMaterial color="#c8a040" />
          </Box>
        )}
      </group>
    </group>
  );
}

function Truck({ startX, endX, moving, alert, label, onPause, color }) {
  const ref = useRef();
  const x = useRef(startX);
  const phase = useRef('loading');
  const timer = useRef(1.0);

  useFrame((_, delta) => {
    if (!ref.current) return;
    if (!moving) {
      if (onPause) onPause(false);
      return;
    }
    if (phase.current === 'loading') {
      timer.current -= delta;
      if (onPause) onPause(true);
      if (timer.current <= 0) {
        phase.current = 'moving';
        timer.current = 2.0;
        if (onPause) onPause(false);
      }
    } else if (phase.current === 'moving') {
      x.current += 0.10;
      if (x.current >= endX) {
        x.current = endX;
        phase.current = 'arriving';
        timer.current = 2.0;
      }
    } else if (phase.current === 'arriving') {
      timer.current -= delta;
      if (timer.current <= 0) {
        phase.current = 'loading';
        timer.current = 1.0;
        x.current = startX;
      }
    }
    ref.current.position.x = x.current;
  });

  return (
    <group ref={ref} position={[startX, -1.15, 0]}>
      <Box args={[1.6, 0.65, 0.85]}>
        <meshStandardMaterial color={alert ? '#cc1100' : color || '#dddddd'} />
      </Box>
      <Box args={[0.5, 0.5, 0.87]} position={[0.65, 0.2, 0]}>
        <meshStandardMaterial color={alert ? '#ff3300' : '#3366cc'} />
      </Box>
      {[[-0.45, -0.38], [0.45, -0.38]].map(([wx, wy], i) => (
        <Cylinder key={i} args={[0.16, 0.16, 0.9, 8]}
          rotation={[Math.PI / 2, 0, 0]} position={[wx, wy, 0]}>
          <meshStandardMaterial color="#111" />
        </Cylinder>
      ))}
      <Text position={[0, 0.6, 0]} fontSize={0.2} color="white" anchorX="center">{label}</Text>
      {alert && <Text position={[0, 0.95, 0]} fontSize={0.22} color="#ff4444" anchorX="center">⛔</Text>}
    </group>
  );
}

function ConveyorBelt({ running, startX, endX }) {
  const refs = useRef([]);
  const pos = useRef([0, 2, 4, 6].map(o => startX + o));
  useFrame(() => {
    if (!running) return;
    pos.current = pos.current.map(x => { const n = x + 0.022; return n > endX ? startX : n; });
    refs.current.forEach((m, i) => { if (m) m.position.x = pos.current[i]; });
  });
  return (
    <group position={[0, -1.52, 0]}>
      <Box args={[endX - startX, 0.1, 0.55]} position={[(startX + endX) / 2, 0, 0]}>
        <meshStandardMaterial color={running ? '#444' : '#1a1a1a'} />
      </Box>
      {Array.from({ length: Math.floor((endX - startX) / 1.1) }, (_, i) => (
        <Cylinder key={i} args={[0.07, 0.07, 0.6, 6]} rotation={[Math.PI / 2, 0, 0]}
          position={[startX + 0.55 + i * 1.1, 0.06, 0]}>
          <meshStandardMaterial color="#666" />
        </Cylinder>
      ))}
      {pos.current.map((_, i) => (
        <Box key={i} ref={el => refs.current[i] = el} args={[0.38, 0.38, 0.45]}
          position={[pos.current[i], 0.27, 0]}>
          <meshStandardMaterial color={running ? '#c8a040' : '#442200'} />
        </Box>
      ))}
    </group>
  );
}

function Scene({ sc }) {
  const [arm1Loading, setArm1Loading] = useState(false);
  const [arm2Loading, setArm2Loading] = useState(false);
  const [armDCLoading, setArmDCLoading] = useState(false);

  return (
    <>
      <ambientLight intensity={0.6} />
      <pointLight position={[0, 10, 5]} intensity={1.2} />
      <OrbitControls enablePan enableZoom />
      <Box args={[55, 0.08, 7]} position={[0, -2.02, 0]}>
        <meshStandardMaterial color="#0d1520" />
      </Box>
      <gridHelper args={[55, 55, '#1a3050', '#0d1a28']} position={[0, -2, 0]} />

      {/* Buildings */}
      <Building position={[-20, -0.5, 0]} color="#2d8b4e" label="Supplier" width={2.5} height={2} />
      <Building position={[-12, -0.2, 0]} color={statusOf(sc, 'WH1').colour}
        label={`WH1: Cotton\n${sc.wh1Inventory}u`} width={3} height={2.4} />
      <Building position={[-3, -0.3, 0]} color={statusOf(sc, 'WH1').colour}
        label="Factory" width={3} height={2.2} />
      <Building position={[5, -0.2, 0]} color={statusOf(sc, 'WH2').colour}
        label={`WH2: Garments\n${sc.wh2Inventory}u`} width={3} height={2.4} />
      <Building position={[14, -0.4, 0]} color={statusOf(sc, 'DC1').colour}
        label={`Dist. Center\n${sc.dcCapacity}%`} width={3.5} height={2.2} />
      <Building position={[22, -0.5, 0]} color="#b8860b"
        label={`Shop\n${sc.retailStock}u`} width={2.5} height={2} />

      {/* Robot Arm at WH1 — stops when WH1 is in an alert state */}
      <RobotArm position={[-10.5, -1.15, 0]}
        active={sc.arm1Running}
        loading={arm1Loading && sc.arm1Running} />

      {/* Robot Arm at Factory — stops on arm failure */}
      <RobotArm position={[-1.2, -1.15, 0]}
        active={sc.arm2Running}
        loading={arm2Loading && sc.arm2Running} />

      {/* Robot Arm at DC — stops when DC full */}
      <RobotArm position={[12.5, -1.15, 0]}
        active={agentOf(sc, 'DC1').status !== 'escalated'}
        loading={armDCLoading} />

      {/* Conveyor Factory → WH2 — stops on arm failure */}
      <ConveyorBelt running={sc.arm2Running} startX={-1} endX={4} />

      {/* Conveyor WH2 → DC — stops on arm failure or DC full */}
      <ConveyorBelt running={sc.arm1Running} startX={7} endX={12} />

      {/* Truck 1: Supplier → WH1 — stops when WH1 is in an alert state */}
      <Truck startX={-19} endX={-14}
        moving={agentOf(sc, 'WH1').status !== 'escalated'}
        alert={agentOf(sc, 'WH1').status === 'escalated'}
        label="T1: Raw Cotton" color="#aaddaa"
        onPause={() => {}} />

      {/* Truck 2: WH1 → Factory */}
      <Truck startX={-11} endX={-5}
        moving={sc.truck2Moving}
        alert={agentOf(sc, 'WH1').status === 'escalated'}
        label="T2: Cotton"
        onPause={(v) => { setArm1Loading(v); setArm2Loading(v); }} />

      {/* Truck 3: DC → Shop — stops when DC full */}
      <Truck startX={15} endX={21}
        moving={agentOf(sc, 'DC1').status !== 'escalated'}
        alert={agentOf(sc, 'DC1').status === 'escalated'}
        label="T3: Garments" color="#ddaaaa"
        onPause={setArmDCLoading} />
    </>
  );
}

export default function App() {
  const [sc, setSc] = useState({
    // Measured values, straight from warehouse_sensors / inventory.
    wh1Humidity: 52, wh1Inventory: 800,
    wh2Humidity: 48, wh2Inventory: 600,
    factoryTemp: 22, dcCapacity: 55, retailStock: 400,
    arm1Running: true, arm2Running: true,
    truck2Moving: true, truck3Moving: true,
    delayedShipments: 0, openAlerts: 0,
    // The agent's verdict per site, straight from the backend. The scene
    // renders these; it does not compute its own alert judgement.
    agent: {},
  });

  const [chartData, setChartData] = useState({
    labels: [],
    datasets: [
      { label: 'WH1 Humidity%', data: [], borderColor: '#4a90d9', tension: 0.4, pointRadius: 2 },
      { label: 'WH2 Humidity%', data: [], borderColor: '#e74c3c', tension: 0.4, pointRadius: 2 },
      { label: 'Factory Temp°C', data: [], borderColor: '#f39c12', tension: 0.4, pointRadius: 2 },
    ],
  });

  const [alerts, setAlerts] = useState([]);
  const [agentLog, setAgentLog] = useState([]);
  const [connected, setConnected] = useState(false);


  // One entry per agent_decisions row. Deduped by id: the server replays recent
  // decisions on every (re)connect, and duplicates must not stack up.
  const addDecision = useCallback((d) => setAgentLog(p => (
    p.some(x => x.id === d.id) ? p : [d, ...p].slice(0, 12)
  )), []);
  const addAlert = useCallback((msg) => setAlerts(p => [`[${new Date().toLocaleTimeString()}] ${msg}`, ...p].slice(0, 6)), []);

  // Demo injection. This is NOT a UI effect: it POSTs to the backend, which
  // writes a real row into `events`. The agent finds it on its next poll and
  // reasons about it exactly as it would a simulated disruption, and whatever
  // it decides comes back over the same WebSocket as every other decision.
  const [injecting, setInjecting] = useState(null);
  // Which decision cards have their full reasoning open.
  const [expanded, setExpanded] = useState({});

  const triggerCrisis = useCallback(async (type) => {
    if (injecting) return;
    setInjecting(type);
    try {
      const res = await fetch(`${API_BASE}/api/demo/inject`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ kind: type }),
      });
      const body = await res.json();
      if (!res.ok) {
        addAlert(`Injection failed: ${body.detail || res.status}`);
      } else {
        const inj = body.injected || {};
        addAlert(`Injected ${inj.event_type} at ${inj.warehouse} (event #${inj.event_id}) — awaiting agent`);
      }
    } catch (e) {
      addAlert(`Injection failed: ${e.message} (is the backend running?)`);
    } finally {
      setInjecting(null);
    }
  }, [injecting, addAlert]);

  // ---- Live feed -------------------------------------------------------
  // One WebSocket carries both the twin state (for the 3D scene) and the
  // agent's decisions (for the AI log). There is no polling and no local
  // detection: nothing in this component decides whether a reading warrants
  // action. That judgement happens in the backend agent and arrives here as
  // reasoning text the model actually wrote.
  useEffect(() => {
    let ws;
    let retry;
    let closed = false;

    const num = (site, metric, fallback) => {
      const v = site?.readings?.[metric];
      return typeof v === 'number' ? parseFloat(v.toFixed(1)) : fallback;
    };

    const applyTwinState = (state) => {
      const wh1 = state.sites?.WH1;
      const wh2 = state.sites?.WH2;
      const dc = state.sites?.DC1;
      const rt = state.sites?.RT1;

      const wh1Hum = num(wh1, 'humidity', 50);
      const wh2Hum = num(wh2, 'humidity', 48);
      const facTemp = num(wh1, 'temperature', 22);
      const dcCap = num(dc, 'capacity_pct', 55);
      const arm1 = num(wh1, 'robot_arm_1_active', 1) === 1;
      const arm2 = num(wh2, 'robot_arm_1_active', 1) === 1;

      setSc(prev => ({
        ...prev,
        wh1Humidity: wh1Hum,
        wh1Inventory: wh1?.inventory ?? prev.wh1Inventory,
        wh2Humidity: wh2Hum,
        wh2Inventory: wh2?.inventory ?? prev.wh2Inventory,
        factoryTemp: facTemp,
        dcCapacity: dcCap,
        retailStock: rt?.inventory ?? prev.retailStock,
        delayedShipments: state.shipments?.delayed ?? 0,
        openAlerts: state.open_alerts ?? 0,
        // Single source of truth for "is this site in trouble": the agent said so.
        agent: Object.fromEntries(Object.entries(state.sites || {}).map(
          ([code, s]) => [code, {
            status: s.agent_status || 'unassessed',
            note: s.agent_note || null,
            severity: s.agent_severity || null,
          }])),
        arm1Active: arm1,
        arm2Active: arm2,
        // NO threshold judgement here any more. Whether a site is "in trouble"
        // comes from state.sites[*].agent_status above -- the agent's actual
        // conclusion. Equipment animation follows the measured arm state, which
        // is a reading rather than a verdict.
        arm1Running: arm1,
        arm2Running: arm2,
      }));

      setChartData(prev => ({
        labels: [...prev.labels.slice(-12), new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' })],
        datasets: [
          { ...prev.datasets[0], data: [...prev.datasets[0].data.slice(-12), wh1Hum] },
          { ...prev.datasets[1], data: [...prev.datasets[1].data.slice(-12), wh2Hum] },
          { ...prev.datasets[2], data: [...prev.datasets[2].data.slice(-12), facTemp] },
        ],
      }));
    };

    const applyDecision = (d) => {
      addDecision(d);
      // An escalation the agent raised is also surfaced in the alerts panel.
      (d.tool_calls || []).forEach(c => {
        if (c.tool === 'escalate_alert' && c.ok) {
          addAlert(`${(c.input?.severity || '').toUpperCase()}: ${c.input?.reason || ''}`);
        }
      });
    };

    const connect = () => {
      ws = new WebSocket(WS_URL);

      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        if (!closed) retry = setTimeout(connect, 3000);
      };
      ws.onerror = () => ws.close();
      ws.onmessage = (ev) => {
        let msg;
        try { msg = JSON.parse(ev.data); } catch { return; }
        if (msg.type === 'twin_state') {
          applyTwinState(msg.state);
        } else if (msg.type === 'agent_decision') {
          applyDecision(msg.decision);
        }
      };
    };

    connect();
    return () => {
      closed = true;
      clearTimeout(retry);
      if (ws) ws.close();
    };
  }, [addDecision, addAlert]);

  const health = calcHealth(sc);
  const hColor = health > 75 ? '#00cc66' : health > 50 ? '#f39c12' : '#ff4444';
  const C = (a) => a ? '#ff4444' : '#00cc66';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: '#07090f', color: 'white', fontFamily: "'Segoe UI',Arial,sans-serif", overflow: 'hidden' }}>

      {/* HEADER */}
      <div style={{ padding: '8px 20px', background: '#0d1117', borderBottom: '1px solid #1e3a5f', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <span style={{ fontSize: '17px', fontWeight: 'bold', color: '#4a90d9' }}>🏭 Contoso Apparel — Supply Chain Digital Twin</span>
          <span style={{ marginLeft: '12px', fontSize: '11px', color: '#555' }}>Cotton · Garments · Retail | Live Digital Twin</span>
        </div>
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <div style={{ background: '#0a0e1a', padding: '4px 12px', borderRadius: '8px', border: `1px solid ${hColor}`, textAlign: 'center' }}>
            <div style={{ fontSize: '9px', color: '#555' }}>CHAIN HEALTH</div>
            <div style={{ fontSize: '20px', fontWeight: 'bold', color: hColor }}>{health}%</div>
          </div>

          <span style={{ color: statusOf(sc, 'WH1').dot, fontSize: '12px' }}>
            {`WH1: ${statusOf(sc, 'WH1').label}`}
          </span>
          <span style={{ color: statusOf(sc, 'DC1').dot, fontSize: '12px' }}>
            {`DC1: ${statusOf(sc, 'DC1').label}`}
          </span>
          <span style={{ color: connected ? '#00cc66' : '#ffaa00', fontSize: '12px' }}>
            {connected ? '● LIVE' : '● SIMULATION'}
          </span>
        </div>
      </div>

      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>

        {/* 3D */}
        <div style={{ flex: 2.2, position: 'relative' }}>
          <Canvas camera={{ position: [1, 5, 22], fov: 62 }} style={{ background: '#07090f' }}>
            <Scene sc={sc} />
          </Canvas>
          <div style={{ position: 'absolute', bottom: '10px', left: '50%', transform: 'translateX(-50%)', background: 'rgba(0,0,0,0.7)', padding: '4px 14px', borderRadius: '20px', fontSize: '11px', color: '#4a90d9', whiteSpace: 'nowrap' }}>
            Supplier → WH1 → Factory → WH2 → Conveyor → DC → Shop
          </div>
        </div>

        {/* RIGHT PANEL */}
        <div style={{ width: '320px', display: 'flex', flexDirection: 'column', gap: '8px', padding: '10px', overflowY: 'auto', background: '#0d1117' }}>

          {/* Demo Buttons */}
          <div style={{ background: '#0a0e1a', border: '1px solid #3a1a3a', borderRadius: '7px', padding: '10px' }}>
            <div style={{ fontSize: '11px', color: '#cc88ff', marginBottom: '8px' }}>🎮 Inject Real Disruption → agent reacts</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px' }}>
              {[
                { label: '🌡️ Factory Temp', type: 'temp', color: '#4a2a1a' },
                { label: '💧 Humidity', type: 'humidity', color: '#1a2a4a' },
                { label: '⚙️ Arm Fail', type: 'armfail', color: '#4a2a1a' },
                { label: '🏢 DC Full', type: 'dc', color: '#3a1a1a' },
              ].map(btn => (
                <button key={btn.type} onClick={() => triggerCrisis(btn.type)} disabled={!!injecting}
                  title="Writes a real row to the events table; the agent reacts on its next poll"
                  style={{ background: injecting ? '#111' : btn.color, border: `1px solid ${injecting ? '#222' : '#555'}`, color: injecting ? '#333' : 'white', padding: '8px', borderRadius: '6px', cursor: injecting ? 'wait' : 'pointer', fontSize: '11px', fontWeight: 'bold' }}>
                  {injecting === btn.type ? 'injecting…' : btn.label}
                </button>
              ))}
            </div>
          </div>

          {/* KPIs */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px' }}>
            {[
              { label: 'WH1 Humidity', value: `${sc.wh1Humidity}%`, alert: agentOf(sc, 'WH1').status === 'escalated', icon: '💧' },
              { label: 'WH2 Humidity', value: `${sc.wh2Humidity}%`, alert: agentOf(sc, 'WH2').status === 'escalated', icon: '💧' },
              { label: 'WH1 Stock', value: `${sc.wh1Inventory}u`, alert: sc.wh1Inventory < 220, icon: '📦' },
              { label: 'WH2 Stock', value: `${sc.wh2Inventory}u`, alert: false, icon: '📦' },
              { label: 'Factory Temp', value: `${sc.factoryTemp}°C`, alert: sc.factoryTemp > 28, icon: '🌡️' },
              { label: 'DC Capacity', value: `${sc.dcCapacity}%`, alert: agentOf(sc, 'DC1').status === 'escalated', icon: '🏢' },
              { label: 'Retail Stock', value: `${sc.retailStock}u`, alert: sc.retailStock < 100, icon: '🏪' },
              { label: 'Delayed Shipments', value: `${sc.delayedShipments}`, alert: sc.delayedShipments > 0, icon: '🚚' },
            ].map((k, i) => (
              <div key={i} style={{ background: '#0a0e1a', border: `1px solid ${k.alert ? '#ff4444' : '#1e3a5f'}`, borderRadius: '7px', padding: '8px 10px' }}>
                <div style={{ fontSize: '10px', color: '#555' }}>{k.icon} {k.label}</div>
                <div style={{ fontSize: '16px', fontWeight: 'bold', color: C(k.alert) }}>{k.value}</div>
              </div>
            ))}
          </div>

          {/* Equipment */}
          <div style={{ background: '#0a0e1a', border: '1px solid #1e3a5f', borderRadius: '7px', padding: '10px' }}>
            <div style={{ fontSize: '11px', color: '#4a90d9', marginBottom: '6px' }}>⚙️ Equipment</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px', fontSize: '11px' }}>
              {[
                { label: 'Arm WH1', ok: sc.arm1Running },
                { label: 'Arm Factory', ok: sc.arm2Running },
                { label: 'Arm DC', ok: agentOf(sc, 'DC1').status !== 'escalated' },
                { label: 'Truck 1', ok: agentOf(sc, 'WH1').status !== 'escalated' },
                { label: 'Truck 2', ok: sc.truck2Moving },
                { label: 'Truck 3', ok: sc.truck3Moving },
                { label: 'Conveyor', ok: sc.arm1Running },
                { label: 'WH1 status', ok: agentOf(sc, 'WH1').status !== 'escalated', a: 'ESCALATED', o: 'OK' },
              ].map((e, i) => (
                <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '3px 6px', background: '#0d1520', borderRadius: '4px' }}>
                  <span style={{ color: '#aaa' }}>{e.label}</span>
                  <span style={{ color: e.ok ? '#00cc66' : '#ff4444', fontWeight: 'bold' }}>
                    {e.ok ? (e.o || '✓ OK') : (e.a || '✗ FAIL')}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Chart */}
          <div style={{ background: '#0a0e1a', border: '1px solid #1e3a5f', borderRadius: '7px', padding: '10px' }}>
            <div style={{ fontSize: '11px', color: '#4a90d9', marginBottom: '6px' }}>📊 Live Telemetry</div>
            <Line data={chartData} options={{
              responsive: true, animation: false,
              plugins: { legend: { labels: { color: '#aaa', font: { size: 9 }, boxWidth: 10 } } },
              scales: {
                x: { ticks: { color: '#444', font: { size: 8 }, maxTicksLimit: 4, maxRotation: 0 }, grid: { color: '#1a2a3a' } },
                y: { ticks: { color: '#444', font: { size: 8 } }, grid: { color: '#1a2a3a' } }
              }
            }} />
          </div>

          {/* AI Log */}
          <div style={{ background: '#0a0e1a', border: '1px solid #2a4a8a', borderRadius: '7px', padding: '10px' }}>
            <div style={{ fontSize: '11px', color: '#7ab3ff', marginBottom: '6px', display: 'flex', justifyContent: 'space-between' }}>
              <span>🤖 Agent Decisions</span>
              {agentLog.length > 0 && agentLog[0].model && (
                <span style={{ color: '#334455', fontSize: '9px' }}>{agentLog[0].model}</span>
              )}
            </div>

            {agentLog.length === 0 && (
              <div style={{ fontSize: '10px', color: '#334455' }}>
                No decisions yet — waiting for the agent's first cycle.
              </div>
            )}

            <div style={{ maxHeight: '260px', overflowY: 'auto' }}>
              {agentLog.map((d, i) => {
                const newest = i === 0;
                const border = d.error ? '#aa3322' : d.no_action ? '#2a3a4a' : '#4a90d9';
                return (
                  <div key={d.id} style={{
                    margin: '0 0 8px 0', paddingLeft: '6px',
                    borderLeft: `2px solid ${newest ? border : '#1a2a3a'}`,
                    opacity: newest ? 1 : 0.65,
                  }}>
                    <div style={{ fontSize: '9px', color: '#445566', display: 'flex', gap: '6px' }}>
                      <span>{d.created_at ? new Date(d.created_at).toLocaleTimeString() : ''}</span>
                      <span>#{d.id}</span>
                      <span>{d.trigger}</span>
                      {d.retrieved_case_count > 0 && (
                        // Two different claims, shown differently on purpose:
                        // cites_memory means the reasoning actually referred to
                        // the cases; otherwise they were merely supplied.
                        <span
                          title={d.cites_memory
                            ? `Reasoning refers to ${d.retrieved_case_count} retrieved past case(s): #${(d.retrieved_case_ids || []).join(', #')}`
                            : `${d.retrieved_case_count} past case(s) were supplied (#${(d.retrieved_case_ids || []).join(', #')}) but the reasoning does not appear to refer to them`}
                          style={{
                            color: d.cites_memory ? '#c8a8ff' : '#55607a',
                            fontWeight: d.cites_memory ? 'bold' : 'normal',
                            cursor: 'help',
                          }}>
                          🧠 {d.retrieved_case_count} case{d.retrieved_case_count > 1 ? 's' : ''}
                          {d.cites_memory ? ' · used' : ' · supplied'}
                        </span>
                      )}
                      {d.latency_ms != null && <span>{(d.latency_ms / 1000).toFixed(1)}s</span>}
                    </div>

                    {/* The model's own words. Never a canned string. */}
                    {d.error ? (
                      <div style={{ fontSize: '10px', color: '#ff8866', margin: '3px 0' }}>
                        Agent unavailable: {d.error}
                      </div>
                    ) : (() => {
                      const full = d.reasoning || (d.no_action ? 'Reviewed conditions; no action needed.' : '');
                      const isOpen = !!expanded[d.id];
                      const long = full.length > 220;
                      return (
                        <div style={{ margin: '3px 0' }}>
                          <div style={{ fontSize: '10px', color: newest ? '#aac8ff' : '#556677', lineHeight: 1.45, whiteSpace: 'pre-wrap' }}>
                            {isOpen || !long ? full : `${full.slice(0, 220)}…`}
                          </div>
                          {long && (
                            <button
                              onClick={() => setExpanded(e => ({ ...e, [d.id]: !isOpen }))}
                              style={{ background: 'none', border: 'none', color: '#4a90d9', cursor: 'pointer', fontSize: '9px', padding: '2px 0' }}>
                              {isOpen ? '▲ show less' : '▼ why? (full reasoning)'}
                            </button>
                          )}
                          {isOpen && (d.tool_results || []).length > 0 && (
                            <div style={{ marginTop: '4px', borderTop: '1px solid #1a2a3a', paddingTop: '4px' }}>
                              {(d.tool_calls || []).map((c, k) => (
                                <div key={k} style={{ fontSize: '9px', color: '#667788', margin: '2px 0' }}>
                                  <span style={{ color: c.ok ? '#7ab3ff' : '#ff8866' }}>{c.tool}</span>
                                  {' '}<code style={{ color: '#445566' }}>{JSON.stringify(c.input)}</code>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      );
                    })()}

                    {(d.tool_calls || []).length > 0 ? (
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginTop: '3px' }}>
                        {d.tool_calls.map((c, j) => (
                          <span key={j} title={JSON.stringify(c.input)} style={{
                            fontSize: '9px', padding: '1px 5px', borderRadius: '3px',
                            background: c.ok ? '#12243a' : '#3a1414',
                            color: c.ok ? '#7ab3ff' : '#ff8866',
                            border: `1px solid ${c.ok ? '#1e3a5f' : '#5a2222'}`,
                          }}>
                            {c.tool}{c.ok ? '' : ' ✕'}
                          </span>
                        ))}
                      </div>
                    ) : !d.error && (
                      <div style={{ fontSize: '9px', color: '#3a4a5a', marginTop: '2px' }}>
                        no action taken
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Alerts */}
          <div style={{ background: '#0a0e1a', border: '1px solid #3a1a1a', borderRadius: '7px', padding: '10px' }}>
            <div style={{ fontSize: '11px', color: '#ff6644', marginBottom: '6px' }}>🚨 Alerts</div>
            {alerts.length === 0
              ? <div style={{ fontSize: '10px', color: '#222' }}>No alerts</div>
              : alerts.map((a, i) => (
                <div key={i} style={{ fontSize: '10px', color: i === 0 ? '#ff8866' : '#442211', margin: '2px 0', borderLeft: '2px solid #ff4444', paddingLeft: '6px' }}>{a}</div>
              ))}
          </div>

        </div>
      </div>
    </div>
  );
}