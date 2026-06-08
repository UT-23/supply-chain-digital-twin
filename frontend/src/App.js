import React, { useEffect, useState, useRef, useCallback } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Box, Cylinder, Text } from '@react-three/drei';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const API_URL = "http://localhost:8000/api/supplychain";

const AI_MSGS = {
  mold: "🤖 Mold detected in WH1. Stopping Truck1 & Arm1. Quarantining cotton batch. Notifying supplier.",
  humidity: "🤖 WH1 humidity critical. Stopping loading operations. Activating dehumidifiers.",
  temp: "🤖 Factory temperature critical. Pausing production. Cooling system activated.",
  shortage: "🤖 WH1 stock low. Auto-ordering 500 units from supplier.",
  armFail: "🤖 Robot arm failure. Stopping factory conveyor. Maintenance alerted.",
  dc: "🤖 DC near full. Stopping Truck3 & DC arm. Rerouting shipments.",
  resolved: "✅ Crisis resolved. All systems back to normal.",
};

function calcHealth(sc) {
  let s = 100;
  if (sc.wh1Alert) s -= 25;
  if (sc.wh1Mold) s -= 20;
  if (!sc.factoryRunning) s -= 20;
  if (!sc.arm1Active) s -= 10;
  if (!sc.conveyorRunning) s -= 10;
  if (sc.dcAlert) s -= 15;
  if (sc.wh1Inventory < 220) s -= 10;
  return Math.max(0, s);
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
      <Building position={[-12, -0.2, 0]} color={sc.wh1Alert ? '#aa1111' : '#1a6faf'}
        label={`WH1: Cotton\n${sc.wh1Inventory}u`} width={3} height={2.4} />
      <Building position={[-3, -0.3, 0]} color={sc.factoryRunning ? '#1a5fa8' : '#883311'}
        label="Factory" width={3} height={2.2} />
      <Building position={[5, -0.2, 0]} color={sc.wh2Alert ? '#aa1111' : '#1a6faf'}
        label={`WH2: Garments\n${sc.wh2Inventory}u`} width={3} height={2.4} />
      <Building position={[14, -0.4, 0]} color={sc.dcAlert ? '#882200' : '#7b2d8b'}
        label={`Dist. Center\n${sc.dcCapacity}%`} width={3.5} height={2.2} />
      <Building position={[22, -0.5, 0]} color="#b8860b"
        label={`Shop\n${sc.retailStock}u`} width={2.5} height={2} />

      {/* Robot Arm at WH1 — stops on mold/humidity */}
      <RobotArm position={[-10.5, -1.15, 0]}
        active={sc.arm1Active && !sc.wh1Alert}
        loading={arm1Loading && !sc.wh1Alert} />

      {/* Robot Arm at Factory — stops on arm failure */}
      <RobotArm position={[-1.2, -1.15, 0]}
        active={sc.arm2Active}
        loading={arm2Loading && sc.arm2Active} />

      {/* Robot Arm at DC — stops when DC full */}
      <RobotArm position={[12.5, -1.15, 0]}
        active={!sc.dcAlert}
        loading={armDCLoading && !sc.dcAlert} />

      {/* Conveyor Factory → WH2 — stops on arm failure */}
      <ConveyorBelt running={sc.factoryRunning && sc.arm2Active} startX={-1} endX={4} />

      {/* Conveyor WH2 → DC — stops on arm failure or DC full */}
      <ConveyorBelt running={sc.conveyorRunning} startX={7} endX={12} />

      {/* Truck 1: Supplier → WH1 — stops on mold/humidity */}
      <Truck startX={-19} endX={-14}
        moving={!sc.wh1Alert}
        alert={sc.wh1Alert}
        label="T1: Raw Cotton" color="#aaddaa"
        onPause={() => {}} />

      {/* Truck 2: WH1 → Factory */}
      <Truck startX={-11} endX={-5}
        moving={sc.truck2Moving}
        alert={sc.wh1Alert}
        label="T2: Cotton"
        onPause={(v) => { setArm1Loading(v); setArm2Loading(v); }} />

      {/* Truck 3: DC → Shop — stops when DC full */}
      <Truck startX={15} endX={21}
        moving={sc.truck3Moving && !sc.dcAlert}
        alert={sc.dcAlert}
        label="T3: Garments" color="#ddaaaa"
        onPause={setArmDCLoading} />
    </>
  );
}

export default function App() {
  const [sc, setSc] = useState({
    wh1Humidity: 52, wh1Inventory: 800, wh1Alert: false, wh1Mold: false,
    wh2Humidity: 48, wh2Inventory: 600, wh2Alert: false,
    factoryRunning: true, factoryTemp: 22,
    truck2Moving: true, truck3Moving: true,
    arm1Active: true, arm2Active: true,
    conveyorRunning: true,
    dcCapacity: 55, dcAlert: false,
    retailStock: 400,
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
  const [agentLog, setAgentLog] = useState(['🟢 All systems normal. Monitoring supply chain.']);
  const [connected, setConnected] = useState(false);
  const [recovering, setRecovering] = useState(false);
  const [recoveryCount, setRecoveryCount] = useState(0);
  const [forcedCrisis, setForcedCrisis] = useState(null);

  const addAgent = useCallback((msg) => setAgentLog(p => [`[${new Date().toLocaleTimeString()}] ${msg}`, ...p].slice(0, 6)), []);
  const addAlert = useCallback((msg) => setAlerts(p => [`[${new Date().toLocaleTimeString()}] ${msg}`, ...p].slice(0, 6)), []);

  const triggerRecovery = useCallback((name) => {
    setRecovering(true);
    setRecoveryCount(10);
    addAgent(`⏳ Fixing: ${name}. Recovery in 10s...`);
    const iv = setInterval(() => {
      setRecoveryCount(p => {
        if (p <= 1) {
          clearInterval(iv);
          setRecovering(false);
          setForcedCrisis(null);
          setSc(p => ({
            ...p,
            wh1Alert: false, wh1Mold: false, wh2Alert: false,
            factoryRunning: true, arm1Active: true, arm2Active: true,
            conveyorRunning: true, dcAlert: false,
            truck2Moving: true, truck3Moving: true,
          }));
          addAgent(AI_MSGS.resolved);
          addAlert('✅ RESOLVED: Crisis contained. Operations resumed.');
          return 0;
        }
        return p - 1;
      });
    }, 1000);
  }, [addAgent, addAlert]);

  const triggerCrisis = useCallback((type) => {
    if (recovering) return;
    setForcedCrisis(type);
    if (type === 'mold') {
      // Stop Truck1, Arm1, Truck2
      setSc(p => ({
        ...p, wh1Alert: true, wh1Mold: true,
        factoryRunning: false, truck2Moving: false, arm1Active: false,
      }));
      addAlert('🦠 MOLD DETECTED in WH1! Truck1 + Arm1 stopped!');
      addAgent(AI_MSGS.mold);
      setTimeout(() => triggerRecovery('Mold Outbreak'), 3000);
    } else if (type === 'humidity') {
      // Stop Truck1, Arm1
      setSc(p => ({
        ...p, wh1Alert: true, wh1Humidity: 75,
        truck2Moving: false, arm1Active: false,
      }));
      addAlert('💧 WH1 HUMIDITY 75%! Truck1 + Arm1 stopped!');
      addAgent(AI_MSGS.humidity);
      setTimeout(() => triggerRecovery('Humidity Spike'), 3000);
    } else if (type === 'armfail') {
      // Stop factory conveyor, arm2
      setSc(p => ({
        ...p, arm1Active: false, arm2Active: false,
        conveyorRunning: false, factoryRunning: false, truck2Moving: false,
      }));
      addAlert('⚙️ ARM FAILURE! Factory conveyor stopped!');
      addAgent(AI_MSGS.armFail);
      setTimeout(() => triggerRecovery('Arm Failure'), 3000);
    } else if (type === 'dc') {
      // Stop Truck3, DC arm
      setSc(p => ({
        ...p, dcAlert: true, dcCapacity: 95,
        conveyorRunning: false, truck3Moving: false,
      }));
      addAlert('🏢 DC OVERFLOW 95%! Truck3 + DC arm stopped!');
      addAgent(AI_MSGS.dc);
      setTimeout(() => triggerRecovery('DC Overflow'), 3000);
    }
  }, [recovering, triggerRecovery, addAgent, addAlert]);

  useEffect(() => {
    const iv = setInterval(async () => {
      if (recovering || forcedCrisis) return;
      try {
        const res = await fetch(API_URL);
        if (!res.ok) throw new Error();
        const api = await res.json();
        setConnected(true);

        const wh1Hum = parseFloat(api.warehouse1.humidity.toFixed(1));
        const wh2Hum = parseFloat(api.warehouse2.humidity.toFixed(1));
        const facTemp = parseFloat(api.warehouse1.temperature.toFixed(1));
        const dcCap = parseFloat(api.distCenter.capacity.toFixed(1));
        const moldRisk = api.extras?.mold_detected ?? false;
        const armFail = !api.arm1.active;

        if (moldRisk) {
          addAlert('🦠 Mold in WH1! Truck1 + Arm1 stopped!');
          addAgent(AI_MSGS.mold);
          setTimeout(() => triggerRecovery('Mold'), 3000);
        } else if (wh1Hum > 68) {
          addAlert(`💧 WH1 Humidity: ${wh1Hum}% — Truck1 + Arm1 stopped!`);
          addAgent(AI_MSGS.humidity);
        } else if (facTemp > 28) {
          addAlert(`🌡️ Factory: ${facTemp}°C`);
          addAgent(AI_MSGS.temp);
        } else if (armFail) {
          addAlert('⚙️ Arm failure! Factory conveyor stopped!');
          addAgent(AI_MSGS.armFail);
          setTimeout(() => triggerRecovery('Arm'), 3000);
        } else if (dcCap > 88) {
          addAlert(`🏢 DC: ${dcCap}% — Truck3 + DC arm stopped!`);
          addAgent(AI_MSGS.dc);
        } else if (api.warehouse1.inventory < 220) {
          addAlert('📦 WH1 low stock!');
          addAgent(AI_MSGS.shortage);
        }

        setSc({
          wh1Humidity: wh1Hum,
          wh1Inventory: api.warehouse1.inventory,
          wh1Alert: wh1Hum > 68 || moldRisk,
          wh1Mold: moldRisk,
          wh2Humidity: wh2Hum,
          wh2Inventory: api.warehouse2.inventory,
          wh2Alert: wh2Hum > 65,
          factoryRunning: !moldRisk && facTemp <= 28,
          factoryTemp: facTemp,
          // Truck1 stops on mold/humidity
          // Truck2 stops on mold/humidity
          truck2Moving: !moldRisk && wh1Hum <= 68,
          // Truck3 stops on DC full
          truck3Moving: dcCap <= 88,
          // Arms stop on respective crises
          arm1Active: !armFail && !moldRisk && wh1Hum <= 68,
          arm2Active: !armFail,
          // Factory conveyor stops on arm fail
          conveyorRunning: dcCap <= 88 && !armFail,
          dcCapacity: dcCap,
          dcAlert: dcCap > 88,
          retailStock: api.extras?.retail_stock ?? 400,
        });

        setChartData(prev => ({
          labels: [...prev.labels.slice(-12), new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' })],
          datasets: [
            { ...prev.datasets[0], data: [...prev.datasets[0].data.slice(-12), wh1Hum] },
            { ...prev.datasets[1], data: [...prev.datasets[1].data.slice(-12), wh2Hum] },
            { ...prev.datasets[2], data: [...prev.datasets[2].data.slice(-12), facTemp] },
          ],
        }));
      } catch { setConnected(false); }
    }, 2500);
    return () => clearInterval(iv);
  }, [recovering, forcedCrisis, addAgent, addAlert, triggerRecovery]);

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
          {recovering && (
            <div style={{ background: '#1a3a1a', border: '1px solid #00cc66', borderRadius: '8px', padding: '4px 10px', fontSize: '12px', color: '#00cc66' }}>
              ⏳ Recovering... {recoveryCount}s
            </div>
          )}
          <span style={{ color: sc.factoryRunning ? '#00cc66' : '#ff4444', fontSize: '12px' }}>
            {sc.factoryRunning ? '🟢 Factory ON' : '🔴 Factory OFF'}
          </span>
          <span style={{ color: sc.conveyorRunning ? '#00cc66' : '#ff4444', fontSize: '12px' }}>
            {sc.conveyorRunning ? '🟢 Conveyor ON' : '🔴 Conveyor OFF'}
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
            <div style={{ fontSize: '11px', color: '#cc88ff', marginBottom: '8px' }}>🎮 Demo — Trigger Crisis</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px' }}>
              {[
                { label: '🦠 Mold', type: 'mold', color: '#4a1a4a' },
                { label: '💧 Humidity', type: 'humidity', color: '#1a2a4a' },
                { label: '⚙️ Arm Fail', type: 'armfail', color: '#4a2a1a' },
                { label: '🏢 DC Full', type: 'dc', color: '#3a1a1a' },
              ].map(btn => (
                <button key={btn.type} onClick={() => triggerCrisis(btn.type)} disabled={recovering}
                  style={{ background: recovering ? '#111' : btn.color, border: `1px solid ${recovering ? '#222' : '#555'}`, color: recovering ? '#333' : 'white', padding: '8px', borderRadius: '6px', cursor: recovering ? 'not-allowed' : 'pointer', fontSize: '11px', fontWeight: 'bold' }}>
                  {btn.label}
                </button>
              ))}
            </div>
          </div>

          {/* KPIs */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px' }}>
            {[
              { label: 'WH1 Humidity', value: `${sc.wh1Humidity}%`, alert: sc.wh1Alert, icon: '💧' },
              { label: 'WH2 Humidity', value: `${sc.wh2Humidity}%`, alert: sc.wh2Alert, icon: '💧' },
              { label: 'WH1 Stock', value: `${sc.wh1Inventory}u`, alert: sc.wh1Inventory < 220, icon: '📦' },
              { label: 'WH2 Stock', value: `${sc.wh2Inventory}u`, alert: false, icon: '📦' },
              { label: 'Factory Temp', value: `${sc.factoryTemp}°C`, alert: sc.factoryTemp > 28, icon: '🌡️' },
              { label: 'DC Capacity', value: `${sc.dcCapacity}%`, alert: sc.dcAlert, icon: '🏢' },
              { label: 'Retail Stock', value: `${sc.retailStock}u`, alert: sc.retailStock < 100, icon: '🏪' },
              { label: 'Mold WH1', value: sc.wh1Mold ? 'DETECTED' : 'Clear', alert: sc.wh1Mold, icon: '🦠' },
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
                { label: 'Arm WH1', ok: sc.arm1Active },
                { label: 'Arm Factory', ok: sc.arm2Active },
                { label: 'Arm DC', ok: !sc.dcAlert },
                { label: 'Truck 1', ok: !sc.wh1Alert },
                { label: 'Truck 2', ok: sc.truck2Moving },
                { label: 'Truck 3', ok: sc.truck3Moving },
                { label: 'Conveyor', ok: sc.conveyorRunning },
                { label: 'Factory', ok: sc.factoryRunning, a: 'STOPPED', o: 'Running' },
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
            <div style={{ fontSize: '11px', color: '#7ab3ff', marginBottom: '6px' }}>🤖 Agentic AI</div>
            {agentLog.map((a, i) => (
              <div key={i} style={{ fontSize: '10px', color: i === 0 ? '#aac8ff' : '#334455', margin: '2px 0', borderLeft: `2px solid ${i === 0 ? '#4a90d9' : '#1a2a3a'}`, paddingLeft: '6px' }}>{a}</div>
            ))}
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