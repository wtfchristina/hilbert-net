import React, { useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Billboard, Line, OrbitControls, Stars, Text } from '@react-three/drei';
import * as THREE from 'three';
import { type LayerSnapshot } from '../quantum/qcnnEngine';
import { type QuantumPhaseLabel } from '../quantum/clusterIsing';

interface QuantumManifoldSceneProps {
  snapshots: LayerSnapshot[];
  numQubits: number;
  h1: number;
  h2: number;
  truePhase: QuantumPhaseLabel;
  predictedPhase: QuantumPhaseLabel;
  autoRotate: boolean;
}

const PHASE_COLORS: Record<QuantumPhaseLabel, string> = {
  SPT_CLUSTER: '#22d3ee',
  PARAMAGNETIC: '#a855f7',
  ANTIFERROMAGNETIC: '#f59e0b',
};

function QubitBlochNode({
  position,
  bloch,
  isReadout,
  qubitIndex,
  accentColor,
}: {
  position: [number, number, number];
  bloch: { x: number; y: number; z: number; radius: number; entropy: number };
  isReadout: boolean;
  qubitIndex: number;
  accentColor: string;
}) {
  const equatorRef = useRef<THREE.Mesh>(null);
  const meridianRef = useRef<THREE.Mesh>(null);

  useFrame((_, delta) => {
    if (equatorRef.current) {
      equatorRef.current.rotation.z += delta * (isReadout ? 0.85 : 0.3);
    }
    if (meridianRef.current) {
      meridianRef.current.rotation.y += delta * (isReadout ? 0.5 : 0.2);
    }
  });

  const sphereRadius = isReadout ? 0.46 : 0.31;
  const vecEnd: [number, number, number] = [
    bloch.x * sphereRadius * 0.9,
    bloch.z * sphereRadius * 0.9,
    -bloch.y * sphereRadius * 0.9,
  ];

  const purityGlow = Math.max(0.25, bloch.radius);

  return (
    <group position={position}>
      {/* Luminous translucent outer Bloch shell */}
      <mesh>
        <sphereGeometry args={[sphereRadius, 28, 28]} />
        <meshStandardMaterial
          color={isReadout ? accentColor : '#0ea5e9'}
          emissive={isReadout ? accentColor : '#0369a1'}
          emissiveIntensity={isReadout ? 0.35 : 0.18}
          transparent
          opacity={isReadout ? 0.24 : 0.14}
          roughness={0.25}
          metalness={0.2}
          depthWrite={false}
        />
      </mesh>

      {/* Inner purity core indicating 1 - S(rho) */}
      <mesh>
        <sphereGeometry args={[sphereRadius * (0.18 + 0.45 * bloch.radius), 16, 16]} />
        <meshStandardMaterial
          color={isReadout ? accentColor : '#38bdf8'}
          emissive={isReadout ? accentColor : '#0284c7'}
          emissiveIntensity={0.65}
          transparent
          opacity={0.22}
          depthWrite={false}
        />
      </mesh>

      {/* Equatorial ring */}
      <mesh ref={equatorRef} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[sphereRadius, 0.009, 10, 40]} />
        <meshBasicMaterial
          color={isReadout ? accentColor : '#38bdf8'}
          transparent
          opacity={isReadout ? 0.85 : 0.45}
        />
      </mesh>

      {/* Orthogonal meridian ring */}
      <mesh ref={meridianRef}>
        <torusGeometry args={[sphereRadius, 0.006, 8, 36]} />
        <meshBasicMaterial
          color={isReadout ? accentColor : '#64748b'}
          transparent
          opacity={0.3}
        />
      </mesh>

      {/* Reduced density matrix Bloch vector r = (<X>, <Y>, <Z>) */}
      <Line
        points={[
          [0, 0, 0],
          vecEnd,
        ]}
        color={isReadout ? '#ffffff' : '#7dd3fc'}
        lineWidth={isReadout ? 3.2 : 2.2}
      />

      {/* Tip marker scaled by purity |r| */}
      <mesh position={vecEnd}>
        <sphereGeometry args={[0.042 + purityGlow * 0.028, 14, 14]} />
        <meshStandardMaterial
          color="#ffffff"
          emissive={isReadout ? accentColor : '#38bdf8'}
          emissiveIntensity={2.2}
        />
      </mesh>

      {/* Qubit index label */}
      <Text
        position={[0, -sphereRadius - 0.13, 0]}
        fontSize={0.12}
        color={isReadout ? accentColor : '#94a3b8'}
        anchorX="center"
        anchorY="top"
      >
        {isReadout ? `READOUT q${qubitIndex}` : `q${qubitIndex}`}
      </Text>
    </group>
  );
}

function PhaseManifoldBaseplate({
  h1,
  h2,
  truePhase,
}: {
  h1: number;
  h2: number;
  truePhase: QuantumPhaseLabel;
}) {
  const size = 5.4;
  const markerX = (h1 / 2 - 0.5) * size;
  const markerZ = (h2 / 2 - 0.5) * size;
  const phaseColor = PHASE_COLORS[truePhase];

  return (
    <group position={[0, -2.55, 0]}>
      <gridHelper args={[size, 14, '#1e293b', '#0f172a']} />

      {/* Quantum critical boundaries: h1 = 1, h2 = 1, and h1 = h2 */}
      <Line
        points={[
          [0, 0.01, -size / 2],
          [0, 0.01, 0],
        ]}
        color="#a855f7"
        lineWidth={1.8}
      />
      <Line
        points={[
          [-size / 2, 0.01, 0],
          [0, 0.01, 0],
        ]}
        color="#f59e0b"
        lineWidth={1.8}
      />
      <Line
        points={[
          [0, 0.01, 0],
          [size / 2, 0.01, size / 2],
        ]}
        color="#64748b"
        lineWidth={1.6}
      />

      <Text
        position={[-size * 0.26, 0.03, -size * 0.26]}
        rotation={[-Math.PI / 2, 0, 0]}
        fontSize={0.15}
        color="#22d3ee"
      >
        Z₂×Z₂ SPT CLUSTER
      </Text>
      <Text
        position={[size * 0.26, 0.03, -size * 0.24]}
        rotation={[-Math.PI / 2, 0, 0]}
        fontSize={0.14}
        color="#a855f7"
      >
        PARAMAGNETIC (h₁)
      </Text>
      <Text
        position={[-size * 0.21, 0.03, size * 0.27]}
        rotation={[-Math.PI / 2, 0, 0]}
        fontSize={0.14}
        color="#f59e0b"
      >
        ANTIFERROMAGNETIC (h₂)
      </Text>

      {/* Active Hamiltonian (h1, h2) beacon */}
      <group position={[markerX, 0.05, markerZ]}>
        <mesh>
          <cylinderGeometry args={[0.045, 0.15, 0.24, 16]} />
          <meshStandardMaterial
            color={phaseColor}
            emissive={phaseColor}
            emissiveIntensity={2.2}
          />
        </mesh>
        <Line
          points={[
            [0, 0, 0],
            [0, 0.72, 0],
          ]}
          color={phaseColor}
          lineWidth={1.8}
          transparent
          opacity={0.6}
        />
      </group>
    </group>
  );
}

function TensorNetworkHierarchy({
  snapshots,
  numQubits,
  predictedPhase,
}: {
  snapshots: LayerSnapshot[];
  numQubits: number;
  predictedPhase: QuantumPhaseLabel;
}) {
  const accentColor = PHASE_COLORS[predictedPhase];

  const layerLayouts = useMemo(() => {
    const totalLayers = Math.max(1, snapshots.length);
    const yBottom = -1.65;
    const yTop = 2.15;
    const yStep = totalLayers > 1 ? (yTop - yBottom) / (totalLayers - 1) : 0;

    return snapshots.map((snap, lIdx) => {
      const y = yBottom + lIdx * yStep;
      const count = snap.activeQubits.length;
      // Center each pooled scale symmetrically to form a true MERA pyramid
      const layerSpan = Math.min(5.8, Math.max(0, (count - 1) * 0.96));
      const positions: Record<number, [number, number, number]> = {};

      snap.activeQubits.forEach((q, idx) => {
        const frac = count > 1 ? idx / (count - 1) - 0.5 : 0;
        positions[q] = [frac * layerSpan, y, 0];
      });

      return { snap, y, positions, layerSpan };
    });
  }, [snapshots, numQubits]);

  return (
    <group>
      {layerLayouts.map(({ snap, y, positions, layerSpan }, lIdx) => {
        const isTopLayer = lIdx === layerLayouts.length - 1;
        const nextLayout = layerLayouts[lIdx + 1];
        const isConvLayer = snap.label.toLowerCase().includes('conv');
        const labelX = -Math.max(1.35, layerSpan * 0.5 + 0.65);

        return (
          <group key={snap.layerIndex}>
            {/* Layer badge positioned neatly to the left of each pyramid tier */}
            <Billboard position={[labelX, y, 0]}>
              <Text
                position={[0, 0, 0]}
                fontSize={0.125}
                color={isTopLayer ? accentColor : '#94a3b8'}
                anchorX="right"
                anchorY="middle"
              >
                {snap.label.toUpperCase()}
              </Text>
            </Billboard>

            {/* Horizontal intra-layer entanglement filaments + unitary gate diamonds */}
            {snap.entanglementLinks.map((link) => {
              const p1 = positions[link.q1];
              const p2 = positions[link.q2];
              if (!p1 || !p2) return null;
              const mid: [number, number, number] = [
                (p1[0] + p2[0]) / 2,
                (p1[1] + p2[1]) / 2,
                0,
              ];

              return (
                <group key={`${lIdx}-${link.q1}-${link.q2}`}>
                  <Line
                    points={[p1, p2]}
                    color={link.strength > 0.42 ? '#22d3ee' : '#38bdf8'}
                    lineWidth={1.2 + link.strength * 2.4}
                    transparent
                    opacity={0.3 + link.strength * 0.6}
                  />
                  {isConvLayer && (
                    <mesh position={mid} rotation={[0, 0, Math.PI / 4]}>
                      <octahedronGeometry args={[0.075, 0]} />
                      <meshStandardMaterial
                        color="#22d3ee"
                        emissive="#06b6d4"
                        emissiveIntensity={1.6}
                      />
                    </mesh>
                  )}
                </group>
              );
            })}

            {/* Vertical MERA renormalization contraction lines to next layer */}
            {nextLayout &&
              snap.activeQubits.map((q) => {
                const start = positions[q];
                const parentQ = nextLayout.positions[q]
                  ? q
                  : nextLayout.snap.activeQubits.reduce((prev, curr) =>
                      Math.abs(curr - q) < Math.abs(prev - q) ? curr : prev
                    );
                const end = nextLayout.positions[parentQ];
                if (!start || !end) return null;
                const isTracedOut = !nextLayout.positions[q];

                return (
                  <Line
                    key={`renorm-${lIdx}-${q}`}
                    points={[start, end]}
                    color={isTracedOut ? '#64748b' : '#38bdf8'}
                    lineWidth={isTracedOut ? 1.1 : 1.9}
                    transparent
                    opacity={isTracedOut ? 0.32 : 0.65}
                  />
                );
              })}

            {/* Active Qubit Bloch Spheres */}
            {snap.activeQubits.map((q) => {
              const pos = positions[q];
              const bloch = snap.blochVectors[q];
              if (!pos || !bloch) return null;
              return (
                <QubitBlochNode
                  key={`node-${lIdx}-${q}`}
                  position={pos}
                  bloch={bloch}
                  isReadout={isTopLayer && snap.activeQubits.length === 1}
                  qubitIndex={q}
                  accentColor={accentColor}
                />
              );
            })}
          </group>
        );
      })}
    </group>
  );
}

export const QuantumManifoldScene: React.FC<QuantumManifoldSceneProps> = ({
  snapshots,
  numQubits,
  h1,
  h2,
  truePhase,
  predictedPhase,
  autoRotate,
}) => {
  return (
    <Canvas camera={{ position: [0, 1.15, 8.6], fov: 44 }} dpr={[1, 2]}>
      <color attach="background" args={['#050811']} />
      <fog attach="fog" args={['#050811', 12, 24]} />

      <ambientLight intensity={0.85} />
      <pointLight position={[5, 8, 6]} intensity={1.8} color="#38bdf8" />
      <pointLight position={[-6, -3, 4]} intensity={1.1} color="#a855f7" />

      <Stars radius={28} depth={18} count={950} factor={2.5} fade speed={0.4} />

      <TensorNetworkHierarchy
        snapshots={snapshots}
        numQubits={numQubits}
        predictedPhase={predictedPhase}
      />

      <PhaseManifoldBaseplate h1={h1} h2={h2} truePhase={truePhase} />

      <OrbitControls
        enablePan={true}
        enableZoom={true}
        minDistance={4.2}
        maxDistance={15}
        autoRotate={autoRotate}
        autoRotateSpeed={0.5}
      />
    </Canvas>
  );
};
