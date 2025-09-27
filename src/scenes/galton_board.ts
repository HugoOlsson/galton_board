import { createBumpMap } from '$renderer/lib/rendering/bumpMaps/noise'
import {
  addBackgroundGradient,
  addHDRI,
  addSceneLighting,
  HDRIs,
  loadHDRIData
} from '$renderer/lib/rendering/lighting3d'
import { AnimatedScene, HotReloadSetting, SpaceSetting } from '$renderer/lib/scene/sceneClass'
import * as THREE from 'three'

const gravity = new THREE.Vector3(0, -9.81, 0)
const ballRadius = 0.2
const pegRadius = 1

const restitution = 0.4 // 0 = no bounce, 1 = perfectly elastic
const slop = 1e-4 // small separation to avoid sticking
const tangentLoss = 0.03 // percent tangential energy loss on bounce

const bumpMap = createBumpMap({
  width: 1000,
  height: 1000,
  noiseAlgorithm: 'random',
  intensity: 1
})

const ballMaterial = new THREE.MeshStandardMaterial({
  color: 0xffffff,
  roughness: 1.0,
  metalness: 0.1,
  bumpMap: bumpMap
})

const pegMaterial = new THREE.MeshStandardMaterial({
  color: 0xffffff,
  roughness: 0.3, // Bit of roughness for realism
  metalness: 0.85, // Very metallic
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

/*
function createBall(radius: number): Ball {
  const geometry = new THREE.SphereGeometry(radius, 64, 64)
  const sphere = new THREE.Mesh(geometry, ballMaterial)

  return {
    mesh: sphere,
    velocity: new THREE.Vector3(),
    radius
  }
}
*/
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
  const triangleDepth = 11
  let layerCount = 1

  const pegsList: Peg[] = []
  const pegsGroup = new THREE.Group()

  const perPegWidth = 4
  const perLayerHeight = (Math.sqrt(3) / 2) * perPegWidth

  for (let iy = 0; iy < triangleDepth; iy++) {
    for (let ix = 0; ix < layerCount; ix++) {
      for (let iz = 0; iz < layerCount; iz++) {
        const peg = createPeg(pegRadius)

        const xShift = (perPegWidth * (layerCount - 1)) / 2
        peg.mesh.position.set(
          ix * perPegWidth - xShift,
          -iy * perLayerHeight,
          iz * perPegWidth - xShift
        )

        pegsList.push(peg)
        pegsGroup.add(peg.mesh)
      }
    }

    layerCount++
  }

  return {
    list: pegsList,
    group: pegsGroup
  }
}

function createGrid(): THREE.GridHelper {
  const size = 20 // total side length
  const divisions = 20 // number of cells per side
  const grid = new THREE.GridHelper(size, divisions, 0xffffff, 0x666666)
  grid.position.y = 0
  grid.material.opacity = 0.7 // make it subtle
  grid.material.transparent = true
  return grid
}

const pegWorld = new THREE.Vector3()
const delta = new THREE.Vector3()
let normal = new THREE.Vector3()
const velocityNormal = new THREE.Vector3()
const tangentialVelocity = new THREE.Vector3()

function resolveBallPeg(ball: Ball, peg: Peg) {
  peg.mesh.getWorldPosition(pegWorld)

  delta.subVectors(ball.mesh.position, pegWorld)
  const rSum = ball.radius + peg.radius
  const distanceSquared = delta.lengthSq()
  const rSumSquared = rSum * rSum

  if (distanceSquared >= rSumSquared) return

  const dist = Math.sqrt(Math.max(distanceSquared, 1e-12))
  normal.copy(delta).multiplyScalar(1 / dist)

  const penetration = rSum - dist + slop
  ball.mesh.position.addScaledVector(normal, penetration)

  const vdotn = ball.velocity.dot(normal)
  if (vdotn < 0) {
    ball.velocity.addScaledVector(normal, -(1 + restitution) * vdotn)

    velocityNormal.copy(normal).multiplyScalar(ball.velocity.dot(normal)) // normal component after reflection
    tangentialVelocity.copy(ball.velocity).sub(velocityNormal)
    ball.velocity.addScaledVector(tangentialVelocity, -tangentLoss)
  }
}

interface CubeGrid {
  group: THREE.Group
  cells: THREE.Mesh[] // one Mesh per cell
  size: number // world size (edge length)
  divisions: number // cells per side
  cellSize: number // derived: size / divisions
  yPlane: number
}

function createCubeGrid({
  size = 40,
  divisions = 20,
  yPlane = 0,
  baseOpacity = 0.45
} = {}): CubeGrid {
  const group = new THREE.Group()
  const cells: THREE.Mesh[] = []
  const cellSize = size / divisions
  const inset = 0.92 // slight shrink so there’s a gap between cells

  // Reuse one geometry; individual materials optional but cheap at this scale
  const geo = new THREE.BoxGeometry(cellSize * inset, 1, cellSize * inset)

  for (let gz = 0; gz < divisions; gz++) {
    for (let gx = 0; gx < divisions; gx++) {
      const mat = new THREE.MeshStandardMaterial({
        color: 0x3399ff,
        transparent: true,
        opacity: baseOpacity,
        roughness: 0.9,
        metalness: 0.0
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
/*
function addBall(dmScene: AnimatedScene) {
  const ball = createBall(ballRadius)
  ball.mesh.position.x = (Math.random() - 0.5) * 0.1
  ball.mesh.position.z = (Math.random() - 0.5) * 0.1
  ball.mesh.position.y = 65
  dmScene.add(ball.mesh)
  balls.push(ball)
}
*/
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

const hdriData = await loadHDRIData(HDRIs.photoStudio1, 2, 1)

export function galtonBoardScene(): AnimatedScene {
  return new AnimatedScene(
    1080,
    1080,
    SpaceSetting.ThreeDim,
    HotReloadSetting.BeginFromCurrent,
    async (dmScene) => {
      await addHDRI(dmScene, hdriData, 0.3)

      addBackgroundGradient({
        scene: dmScene,
        topColor: 0x0c8ccd, // blue-ish
        bottomColor: 0x000000, // black
        lightingIntensity: 10,
        backgroundOpacity: 0.5
      })

      const pool = new BallPool(dmScene.scene, {
        capacity: 3000, // total preallocated
        radius: ballRadius,
        material: ballMaterial, // your material
        spawnY: 80
      })

      const targetActive = 250 // simulate at most this many at once
      const spawnPerSec = 120 // steady inflow
      let spawnAccum = 0

      const yKill = -2 // when below this -> retire

      addSceneLighting(dmScene.scene)

      const pegsSetup = createPegSetup()
      pegsSetup.group.position.y = 75

      dmScene.add(pegsSetup.group)

      const cubeGrid = createCubeGrid({ size: 140, divisions: 65, yPlane: 0, baseOpacity: 0.8 })
      dmScene.add(cubeGrid.group)

      dmScene.camera.position.set(1.234074, 45.59445, 112.5189)

      dmScene.camera.quaternion.set(-0.1713081, 0.003481703, 0.000605397, 0.9852112)

      const LOOK_AT = new THREE.Vector3(0, 0, 0)
      const baseRadius = dmScene.camera.position.distanceTo(LOOK_AT)
      let angle = 0

      let lastTickTime = 0
      dmScene.onEachTick((tick, time) => {
        const dt = (time - lastTickTime) / 1000
        lastTickTime = time

        spawnAccum += spawnPerSec * dt
        const want = Math.min(targetActive - pool.active.length, Math.floor(spawnAccum))
        for (let i = 0; i < want; i++) pool.spawn()
        spawnAccum -= want

        let i = 0
        while (i < pool.active.length) {
          const b = pool.active[i]
          const prevY = b.prevY

          // integrate
          b.velocity.addScaledVector(gravity, dt)
          b.mesh.position.addScaledVector(b.velocity, dt)

          // collide with pegs
          for (const peg of pegsSetup.list) {
            resolveBallPeg(b, peg)
          }

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

        angle += 0.005
        dmScene.camera.position.x = Math.sin(angle) * baseRadius
        dmScene.camera.position.z = Math.cos(angle) * baseRadius
        dmScene.camera.lookAt(LOOK_AT)
      })

      dmScene.addWait(5000_000)
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
  const colorA = new THREE.Color(0x0000ff) // Blue
  const colorB = new THREE.Color(0xff0000) // Red

  // Normalize h to 0-1 range (adjust min/max based on your expected h range)
  const minH = 0.001
  const maxH = 20.0 // adjust based on your max expected height
  const t = Math.min(1, Math.max(0, (h - minH) / (maxH - minH)))

  // Interpolate between colors
  ;(m.material as any).color = new THREE.Color().lerpColors(colorA, colorB, t)
}
