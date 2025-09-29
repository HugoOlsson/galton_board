import { createBumpMap } from '$renderer/lib/rendering/bumpMaps/noise'
import {
  addBackgroundGradient,
  addHDRI,
  addSceneLighting,
  HDRIs,
  loadHDRIData
} from '$renderer/lib/rendering/lighting3d'
import { AnimatedScene, HotReloadSetting, SpaceSetting } from '$renderer/lib/scene/sceneClass'
import { createFastText, updateText } from '$renderer/lib/rendering/objects2d'
import * as THREE from 'three'
import { moveCameraAnimation3D, rotateCamera3D } from '$renderer/lib/animation/animations'
import { createAnim } from '$renderer/lib/animation/protocols'
import { easeInOutQuad } from '$renderer/lib/animation/interpolations'

const gravity = new THREE.Vector3(0, -9.81, 0)
const ballRadius = 0.27
const pegRadius = 1
const tilesStartColor = new THREE.Color(0x808080)

let collisionCount = 0
let counterText: any
let lastShown = -1

const TIME_SPEED = 2.5

const restitution = 0.3 // 0 = no bounce, 1 = perfectly elastic

const PER_PEG_XZ = 4
const PER_LAYER_Y = (Math.sqrt(3) / 2) * PER_PEG_XZ

const MAX_POOL_CAPACITY = 50_000
const TARGET_ACTIVE = 15_000
const SPAWN_PER_SEC = 7500

const bumpMap = createBumpMap({
  width: 1000,
  height: 1000,
  noiseAlgorithm: 'random',
  intensity: 1
})

const ballMaterial = new THREE.MeshStandardMaterial({
  color: 0x333333,
  roughness: 0.5,
  metalness: 1,
  bumpMap: bumpMap
})

const pegMaterial = new THREE.MeshStandardMaterial({
  color: 0x808080,
  roughness: 0.3, // Bit of roughness for realism
  metalness: 1, // Very metallic
  bumpMap: bumpMap
})

interface Ball {
  mesh: THREE.Mesh
  velocity: THREE.Vector3
  radius: number
  prevY: number
  activeIndex: number
}

interface Peg {
  mesh: THREE.Mesh
  radius: number
}

function createPeg(radius: number): Peg {
  const geometry = new THREE.SphereGeometry(radius, 64, 64)
  const sphere = new THREE.Mesh(geometry, pegMaterial)

  return {
    mesh: sphere,
    radius
  }
}

interface PegSetup {
  list: Peg[]
  group: THREE.Group
}

function createPegSetup(): PegSetup {
  const depth = 20
  let layerCount = 1
  const pegsList: Peg[] = []
  const pegsGroup = new THREE.Group()

  const s = PER_PEG_XZ
  const yStep = PER_LAYER_Y
  const a = 0.05 * s // use your varianceFactor*s if you like
  const jitterAmp = 0.02 * s

  for (let iy = 0; iy < depth; iy++) {
    const xShift = (layerCount - 1) * s * 0.5
    const zShift = (layerCount - 1) * s * 0.5

    // First pass: generate positions for this layer
    type P = { peg: Peg; x: number; z: number; y: number }
    const layer: P[] = []
    let sumX = 0,
      sumZ = 0

    for (let iz = 0; iz < layerCount; iz++) {
      // hex-like: alternate rows in X, zero-mean (±a)
      const rowShiftX = iz & 1 ? +a : -a

      for (let ix = 0; ix < layerCount; ix++) {
        const peg = createPeg(pegRadius)
        peg.mesh.matrixAutoUpdate = false

        // base grid
        let x = ix * s - xShift + rowShiftX
        let z = iz * s - zShift

        // small parity-based micro-shifts with zero mean inside the layer
        // (avoid repeating corridors without introducing drift)
        const microX = iy & 1 ? (ix & 1 ? +0.5 * a : -0.5 * a) : 0
        const microZ = iy & 2 ? (iz & 1 ? +0.5 * a : -0.5 * a) : 0
        x += microX
        z += microZ

        // tiny zero-mean jitter
        x += (Math.random() - 0.5) * jitterAmp
        z += (Math.random() - 0.5) * jitterAmp

        const y = -iy * yStep

        layer.push({ peg, x, z, y })
        sumX += x
        sumZ += z
      }
    }

    // Center-correct this layer exactly
    const n = layer.length
    const meanX = sumX / n
    const meanZ = sumZ / n

    // Second pass: place pegs with mean removed
    for (const p of layer) {
      p.peg.mesh.position.set(p.x - meanX, p.y, p.z - meanZ)
      p.peg.mesh.updateMatrix()
      pegsList.push(p.peg)
      pegsGroup.add(p.peg.mesh)
    }

    layerCount++
  }

  return { list: pegsList, group: pegsGroup }
}

interface CubeGrid {
  group: THREE.Group
  cells: THREE.Mesh[] // one Mesh per cell
  size: number // world size (edge length)
  divisions: number // cells per side
  cellSize: number // derived: size / divisions
  yPlane: number
}

function createCubeGrid({ size = 40, divisions = 20, yPlane = 0 } = {}): CubeGrid {
  const group = new THREE.Group()
  const cells: THREE.Mesh[] = []
  const cellSize = size / divisions
  const inset = 0.95 // slight shrink so there’s a gap between cells

  // Reuse one geometry; individual materials optional but cheap at this scale
  const geo = new THREE.BoxGeometry(cellSize * inset, 1, cellSize * inset)

  for (let gz = 0; gz < divisions; gz++) {
    for (let gx = 0; gx < divisions; gx++) {
      const mat = new THREE.MeshStandardMaterial({
        color: tilesStartColor,
        transparent: false,
        roughness: 0.8,
        metalness: 0
      })

      const m = new THREE.Mesh(geo, mat)
      // Start essentially flat
      m.scale.y = 0.001

      // Centered grid on (0, yPlane, 0)
      const x = (gx + 0.5) * cellSize - size / 2
      const z = (gz + 0.5) * cellSize - size / 2
      m.position.set(x, yPlane + m.scale.y * 0.5, z)

      group.add(m)
      cells.push(m)
    }
  }

  return { group, cells, size, divisions, cellSize, yPlane }
}

class BallPool {
  private inactive: Ball[] = []
  public active: Ball[] = [] // dense list; removal uses swap-pop
  private scene: THREE.Scene
  private ballGeo: THREE.SphereGeometry
  private ballMat: THREE.Material
  private spawnY: number

  constructor(
    scene: THREE.Scene,
    {
      capacity = 500,
      radius = 0.2,
      material,
      geometry,
      spawnY = 65
    }: {
      capacity?: number
      radius?: number
      material: THREE.Material
      geometry?: THREE.SphereGeometry
      spawnY?: number
    }
  ) {
    this.scene = scene
    this.spawnY = spawnY
    this.ballGeo = geometry ?? new THREE.SphereGeometry(radius, 32, 16)
    this.ballMat = material

    for (let i = 0; i < capacity; i++) {
      const mesh = new THREE.Mesh(this.ballGeo, this.ballMat)
      mesh.visible = false
      const b: Ball = {
        mesh,
        velocity: new THREE.Vector3(),
        radius,
        prevY: Number.POSITIVE_INFINITY,
        activeIndex: -1
      }
      this.inactive.push(b)
      this.scene.add(mesh)
    }
  }

  spawn(setup?: (b: Ball) => void): Ball | null {
    const b = this.inactive.pop()
    if (!b) return null
    // reset state
    b.velocity.set(0, 0, 0)
    b.prevY = Number.POSITIVE_INFINITY
    b.mesh.position.set((Math.random() - 0.5) * 0.1, this.spawnY, (Math.random() - 0.5) * 0.1)
    b.mesh.visible = true
    if (setup) setup(b)

    b.activeIndex = this.active.length
    this.active.push(b)
    return b
  }

  retireAt(index: number) {
    // swap-and-pop from `active`
    const last = this.active.length - 1
    const b = this.active[index]
    const swapB = this.active[last]
    this.active[index] = swapB
    swapB.activeIndex = index
    this.active.pop()

    // deactivate b
    b.mesh.visible = false
    b.activeIndex = -1
    this.inactive.push(b)
  }

  retire(b: Ball) {
    if (b.activeIndex >= 0) this.retireAt(b.activeIndex)
  }
}

const hdriData = await loadHDRIData(HDRIs.photoStudio2, 2, 1)

export function galtonBoardScene(): AnimatedScene {
  return new AnimatedScene(
    2000,
    2000,
    SpaceSetting.ThreeDim,
    HotReloadSetting.BeginFromCurrent,
    async (dmScene) => {
      await addHDRI(dmScene, hdriData, 0.3)

      addBackgroundGradient({
        scene: dmScene,
        topColor: 0xffffff,
        bottomColor: 0xffffff,
        lightingIntensity: 10,
        backgroundOpacity: 1
      })

      const pool = new BallPool(dmScene.scene, {
        capacity: MAX_POOL_CAPACITY, // total preallocated
        radius: ballRadius,
        material: ballMaterial, // your material
        spawnY: 100
      })

      let spawnAccum = 0

      const yKill = -ballRadius * 2 // when below this -> retire

      addSceneLighting(dmScene.scene, { colorScheme: 'studio' })

      // --- pegs ---
      const pegsSetup = createPegSetup()
      pegsSetup.group.position.y = 95
      dmScene.add(pegsSetup.group)

      // Build broad-phase once (cell sizes ≈ peg spacing)
      pegsSetup.group.updateMatrixWorld(true)
      const pegIndex = buildPegSpatialIndex(pegsSetup, {
        cellX: PER_PEG_XZ,
        cellY: PER_LAYER_Y,
        cellZ: PER_PEG_XZ
      })

      const cubeGrid = createCubeGrid({ size: 140, divisions: 53, yPlane: 0 })
      dmScene.add(cubeGrid.group)

      // Camera starting position
      dmScene.camera.position.set(-0.03595941, 34.94228, 163.5067)

      dmScene.camera.quaternion.set(0.02028413, -0.002545129, 0.00005163652, 0.999791)

      const LOOK_AT = new THREE.Vector3(0, 40, 0)
      const baseRadius = dmScene.camera.position.distanceTo(LOOK_AT)
      let angle = 0

      // HUD holder that rides with the camera
      const hud = new THREE.Group()
      dmScene.add(hud)

      // Place in camera space: x=left/right, y=up/down, z<0 is in front of camera
      hud.position.set(0, 105, 0)

      // Create the 3D text
      counterText = await createFastText('Collisions: 0', 6.5, 0x000000)

      hud.add(counterText)

      let lastTickTime = 0
      dmScene.onEachTick((tick, time) => {
        const dt = (TIME_SPEED * (time - lastTickTime)) / 1000
        lastTickTime = time

        spawnAccum += SPAWN_PER_SEC * dt
        const want = Math.min(TARGET_ACTIVE - pool.active.length, Math.floor(spawnAccum))
        for (let i = 0; i < want; i++) pool.spawn()
        spawnAccum -= want

        let i = 0
        while (i < pool.active.length) {
          const b = pool.active[i]
          const prevY = b.prevY

          // integrate
          b.velocity.addScaledVector(gravity, dt)
          b.mesh.position.addScaledVector(b.velocity, dt)

          // collide with nearby pegs (numeric-hash broad-phase)
          resolveBallPegSpatial(b, pegIndex, () => {
            collisionCount++
          })

          // count downward crossing of the tiles plane using the bottom of the ball
          const plane = cubeGrid.yPlane
          const y0 = prevY - b.radius
          const y1 = b.mesh.position.y - b.radius
          if (y0 > plane && y1 <= plane && b.velocity.y < 0) {
            const idx = gridIndexFromXZ(cubeGrid, b.mesh.position.x, b.mesh.position.z)
            incrementCell(cubeGrid, idx, 0.1)
          }

          // retire below kill plane
          if (b.mesh.position.y < yKill) {
            pool.retireAt(i) // swap-pop, don't increment i
            continue
          }

          // now update prevY for the next frame
          b.prevY = b.mesh.position.y
          i++
        }

        if (collisionCount !== lastShown) {
          lastShown = collisionCount
          updateText(counterText, `Collisions: ${collisionCount.toLocaleString()}`)
        }
        hud.lookAt(dmScene.camera.position)
      })

      const rotationAnimation = createAnim(easeInOutQuad(0, 2 * Math.PI, 9000), (value) => {
        dmScene.camera.position.x = Math.sin(value) * baseRadius
        dmScene.camera.position.z = Math.cos(value) * baseRadius
        dmScene.camera.lookAt(LOOK_AT)
      })

      dmScene.addAnim(rotationAnimation)

      const MDAnimationTime = 1500
      dmScene.do((startTick) => {
        const endPos = new THREE.Vector3(0, -150, 0)

        const moveAnimation = moveCameraAnimation3D(
          dmScene.camera,
          dmScene.camera.position.clone(),
          endPos,
          MDAnimationTime
        )

        const baseUpdater = moveAnimation.updater

        moveAnimation.updater = (interp, sceneTick, isLast) => {
          // 1) do the position update
          baseUpdater(interp, sceneTick, isLast)
          dmScene.camera.lookAt(LOOK_AT)
        }

        dmScene.insertAnimAt(startTick, moveAnimation)
      })

      dmScene.addWait(MDAnimationTime)

      // Stay to look at the bottom
      dmScene.addWait(6000)

      // Move back
      const MBAnimationTime = 1500
      dmScene.do((startTick) => {
        const endPos = new THREE.Vector3(-0.03595941, 34.94228, 163.5067)

        const moveAnimation = moveCameraAnimation3D(
          dmScene.camera,
          dmScene.camera.position.clone(),
          endPos,
          MBAnimationTime
        )

        const baseUpdater = moveAnimation.updater

        moveAnimation.updater = (interp, sceneTick, isLast) => {
          // 1) do the position update
          baseUpdater(interp, sceneTick, isLast)
          dmScene.camera.lookAt(LOOK_AT)
        }

        dmScene.insertAnimAt(startTick, moveAnimation)
      })

      dmScene.addWait(MBAnimationTime)

      dmScene.addAnim(rotationAnimation.copy().scaleLength(1.25))
    }
  )
}

function gridIndexFromXZ(grid: CubeGrid, x: number, z: number): number {
  const { size, divisions } = grid
  const u = (x + size / 2) / size
  const v = (z + size / 2) / size
  const gx = Math.floor(u * divisions)
  const gz = Math.floor(v * divisions)
  if (gx < 0 || gx >= divisions || gz < 0 || gz >= divisions) return -1
  return gz * divisions + gx
}

function incrementCell(grid: CubeGrid, index: number, dh = 0.3) {
  if (index < 0) return
  const m = grid.cells[index]
  const h = Math.max(0.001, m.scale.y + dh)
  m.scale.y = h
  m.position.y = grid.yPlane + h * 0.5

  // Define your two colors
  const colorA = new THREE.Color(0x000000)
  const colorB = new THREE.Color(0x054afa)

  // Normalize h to 0-1 range (adjust min/max based on your expected h range)
  const minH = 0.001
  const maxH = 10.0 // adjust based on your max expected height
  const t = Math.min(1, Math.max(0, (h - minH) / (maxH - minH)))

  // Interpolate between colors
  ;(m.material as any).color = new THREE.Color().lerpColors(colorA, colorB, t)
}

// ---- Numeric key packing (solution 1) ----
const BASE = 1 << 17 // 131,072
const OFF = BASE >> 1 // bias to support negatives
const B2 = BASE * BASE // stride for +1 in X

const packKey = (ix: number, iy: number, iz: number) =>
  ((ix + OFF) * BASE + (iy + OFF)) * BASE + (iz + OFF)

// Precompute the 27 neighbor offsets (dx * B2 + dy * BASE + dz)
const NEIGHBOR_OFFS: number[] = []
for (let dz = -1; dz <= 1; dz++)
  for (let dy = -1; dy <= 1; dy++)
    for (let dx = -1; dx <= 1; dx++) NEIGHBOR_OFFS.push(dx * B2 + dy * BASE + dz)

interface PegSpatialIndex {
  px: Float32Array
  py: Float32Array
  pz: Float32Array // world centers
  buckets: Map<number, number[]> // packedKey -> peg indices
  cellX: number
  cellY: number
  cellZ: number
  count: number
}

function buildPegSpatialIndex(
  setup: PegSetup,
  { cellX, cellY, cellZ }: { cellX: number; cellY: number; cellZ: number }
): PegSpatialIndex {
  // Make sure world matrices reflect group transforms
  setup.group.updateMatrixWorld(true)

  const n = setup.list.length
  const px = new Float32Array(n)
  const py = new Float32Array(n)
  const pz = new Float32Array(n)
  const buckets = new Map<number, number[]>()

  for (let i = 0; i < n; i++) {
    // Read world position from matrixWorld directly (no allocations)
    const e = setup.list[i].mesh.matrixWorld.elements
    const x = e[12],
      y = e[13],
      z = e[14]
    px[i] = x
    py[i] = y
    pz[i] = z

    const ix = Math.floor(x / cellX)
    const iy = Math.floor(y / cellY)
    const iz = Math.floor(z / cellZ)
    const key = packKey(ix, iy, iz)

    let arr = buckets.get(key)
    if (!arr) {
      arr = []
      buckets.set(key, arr)
    }
    arr.push(i)
  }

  return { px, py, pz, buckets, cellX, cellY, cellZ, count: n }
}

// Precompute collision constants (radii are constants)
const RSUM = ballRadius + pegRadius
const RSUM2 = RSUM * RSUM

function resolveBallPegSpatial(
  ball: Ball,
  idx: PegSpatialIndex,
  onCollide?: (pegIndex: number) => void
): boolean {
  // Tunables (local to keep this self-contained)
  const PEN_SLOP = 1e-3 // ignore tiny interpenetrations
  const POS_CORRECTION_FRACTION = 0.8 // 0..1 split-impulse push fraction
  const REST_VEL_THRESHOLD = 0.2 // below => treat as resting (no bounce)
  const MU_K_DEFAULT = 0.12 // fallback kinetic friction
  const MICRO_BIAS = 1e-4 // tiny extra depenetration to avoid re-hit

  const pegRest = (idx as any).pegRest as Float32Array | undefined
  const pegMu = (idx as any).pegMu as Float32Array | undefined

  const p = ball.mesh.position
  const v = ball.velocity

  const bx = p.x,
    by = p.y,
    bz = p.z

  const ix = Math.floor(bx / idx.cellX)
  const iy = Math.floor(by / idx.cellY)
  const iz = Math.floor(bz / idx.cellZ)
  const baseKey = packKey(ix, iy, iz)

  let hitAny = false

  // Check 27 neighbor buckets
  for (let n = 0; n < 27; n++) {
    const arr = idx.buckets.get(baseKey + NEIGHBOR_OFFS[n])
    if (!arr) continue

    for (let k = 0; k < arr.length; k++) {
      const j = arr[k]

      // delta = ball - peg
      const dx = bx - idx.px[j]
      const dy = by - idx.py[j]
      const dz = bz - idx.pz[j]
      const d2 = dx * dx + dy * dy + dz * dz
      if (d2 >= RSUM2) continue

      // Contact normal
      const dist = Math.sqrt(Math.max(d2, 1e-12))
      const inv = 1 / dist
      const nx = dx * inv,
        ny = dy * inv,
        nz = dz * inv

      // --- Position correction (split impulse, zero-energy) ---
      const rawPen = RSUM - dist // desired pushout
      const corr = Math.max(0, rawPen - PEN_SLOP) // ignore small
      if (corr > 0) {
        const push = POS_CORRECTION_FRACTION * corr
        p.x += nx * push
        p.y += ny * push
        p.z += nz * push
        // micro bias to avoid immediate re-collision
        p.x += nx * MICRO_BIAS
        p.y += ny * MICRO_BIAS
        p.z += nz * MICRO_BIAS
        hitAny = true
      }

      // --- Velocity response (bounce + Coulomb friction) ---
      const vdotn = v.x * nx + v.y * ny + v.z * nz
      if (vdotn < 0) {
        // Restitution: zero if near-rest impact
        const baseRest = pegRest ? pegRest[j] : typeof restitution === 'number' ? restitution : 0.3
        const eEff = -vdotn > REST_VEL_THRESHOLD ? baseRest : 0.0

        // Normal "impulse" (unit mass → delta-v)
        const Jn = -(1 + eEff) * vdotn
        v.x += nx * Jn
        v.y += ny * Jn
        v.z += nz * Jn

        // Tangential (Coulomb) friction
        const vdotn2 = v.x * nx + v.y * ny + v.z * nz
        let tx = v.x - vdotn2 * nx
        let ty = v.y - vdotn2 * ny
        let tz = v.z - vdotn2 * nz
        const vt = Math.hypot(tx, ty, tz)
        if (vt > 1e-8) {
          const invVt = 1 / vt
          tx *= invVt
          ty *= invVt
          tz *= invVt // tangential unit vector
          const mu = pegMu ? pegMu[j] : MU_K_DEFAULT
          const Jt = Math.min(mu * Math.abs(Jn), vt) // clamp to available tangential speed
          v.x -= tx * Jt
          v.y -= ty * Jt
          v.z -= tz * Jt
        }

        if (onCollide) onCollide(j)
        hitAny = true
      }
    }
  }

  return hitAny
}
