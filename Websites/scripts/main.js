const canvas = document.createElement('canvas');
const context = canvas.getContext('2d');
canvas.width = 256;
canvas.height = 64;
context.fillStyle = '#8195ab';
context.font = '12px Arial';
context.textAlign = 'center';

// Ensure `node` is defined before using it
function createNodeLabel(node) {
    context.fillText(node.id, 128, 32);

    const texture = new THREE.CanvasTexture(canvas);
    const labelMaterial = new THREE.SpriteMaterial({
        map: texture,
        transparent: true
    });
    const label = new THREE.Sprite(labelMaterial);
    label.position.y = 0.6;
    label.scale.set(2, 0.5, 1);

    return label;
}

// Example usage of createNodeLabel
const node = { id: 'node1' }; 
const label = createNodeLabel(node);
const nodeMesh = new THREE.Mesh(
    new THREE.SphereGeometry(0.2, 32, 32),
    new THREE.MeshStandardMaterial({ color: 0xffffff })
);
const networkGraph = {
    nodes: {
        node1: { id: 'node1', position: { x: 0, y: 0, z: 0 }, asset_value: 5, vulnerability: 0.2 },
        node2: { id: 'node2', position: { x: 1, y: 1, z: 0 }, asset_value: 7, vulnerability: 0.5 },
        node3: { id: 'node3', position: { x: -1, y: -1, z: 0 }, asset_value: 3, vulnerability: 0.1 },
    },
    edges: [
        { source: 'node1', target: 'node2', vulnerability: 0.3 },
        { source: 'node2', target: 'node3', vulnerability: 0.4 },
    ],
};
nodeMesh.add(label);

networkGraph.edges.forEach(edge => {
    const sourceNode = networkGraph.nodes[edge.source];
    const targetNode = networkGraph.nodes[edge.target];

    const sourcePos = new THREE.Vector3(sourceNode.position.x, sourceNode.position.y, sourceNode.position.z);
    const targetPos = new THREE.Vector3(targetNode.position.x, targetNode.position.y, targetNode.position.z);

    const edgeGeometry = new THREE.BufferGeometry().setFromPoints([sourcePos, targetPos]);

    const edgeMaterial = new THREE.LineDashedMaterial({
        color: 0x4fc3f7,
        linewidth: 1,
        opacity: 0.7,
        transparent: true,
        dashSize: 0.3,
        gapSize: 0.1
    });

    const edge3D = new THREE.Line(edgeGeometry, edgeMaterial);
    edge3D.computeLineDistances();
    edge3D.userData = {
        source: edge.source,
        target: edge.target,
        vulnerability: edge.vulnerability
    };

    scene.add(edge3D);
    edgeObjects.push(edge3D);
});

if (showAttackPaths) {
    createAttackPathVisuals();
}

updateStats();

function clearScene() {
    Object.values(nodeObjects).forEach(node => {
        scene.remove(node);
    });

    edgeObjects.forEach(edge => {
        scene.remove(edge);
    });

    pathObjects.forEach(path => {
        scene.remove(path);
    });

    nodeObjects = {};
    edgeObjects = [];
    pathObjects = [];
}

function createAttackPathVisuals() {
    pathObjects.forEach(path => {
        scene.remove(path);
    });
    pathObjects = [];

    if (!showAttackPaths) return;

    const pathMaterial = new THREE.LineBasicMaterial({
        color: 0xffab40,
        linewidth: 2,
        transparent: true,
        opacity: 0.8
    });

    attackPaths.forEach(path => {
        const points = [];

        for (let i = 0; i < path.length - 1; i++) {
            const startNode = networkGraph.nodes[path[i]];
            const endNode = networkGraph.nodes[path[i + 1]];

            const start = new THREE.Vector3(startNode.position.x, startNode.position.y, startNode.position.z);
            const end = new THREE.Vector3(endNode.position.x, endNode.position.y, endNode.position.z);

            const midPoint = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
            midPoint.y += 0.5;

            const curve = new THREE.QuadraticBezierCurve3(
                start,
                midPoint,
                end
            );

            const curvePoints = curve.getPoints(10);
            points.push(...curvePoints);
        }

        const pathGeometry = new THREE.BufferGeometry().setFromPoints(points);
        const pathLine = new THREE.Line(pathGeometry, pathMaterial);

        const particleGeometry = new THREE.BufferGeometry();
        const particleCount = 5;
        const particlePositions = new Float32Array(particleCount * 3);

        for (let i = 0; i < particleCount; i++) {
            const position = points[0];
            particlePositions[i * 3] = position.x;
            particlePositions[i * 3 + 1] = position.y;
            particlePositions[i * 3 + 2] = position.z;
        }

        particleGeometry.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3));

        const particleMaterial = new THREE.PointsMaterial({
            color: 0xffcc80,
            size: 0.3,
            transparent: true,
            opacity: 0.8,
            map: createParticleTexture(),
            blending: THREE.AdditiveBlending
        });

        const particles = new THREE.Points(particleGeometry, particleMaterial);

        particles.userData = {
            points: points,
            particlePositions: particlePositions,
            particleProgress: Array(particleCount).fill(0).map(() => Math.random())
        };

        scene.add(pathLine);
        scene.add(particles);

        pathObjects.push(pathLine);
        pathObjects.push(particles);

        animateParticles(particles);
    });
}

function createParticleTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 32;
    canvas.height = 32;
    const context = canvas.getContext('2d');

    const gradient = context.createRadialGradient(16, 16, 0, 16, 16, 16);
    gradient.addColorStop(0, 'rgba(255, 255, 255, 1)');
    gradient.addColorStop(0.3, 'rgba(255, 204, 128, 0.8)');
    gradient.addColorStop(0.7, 'rgba(255, 171, 64, 0.4)');
    gradient.addColorStop(1, 'rgba(255, 171, 64, 0)');

    context.fillStyle = gradient;
    context.fillRect(0, 0, 32, 32);

    return new THREE.CanvasTexture(canvas);
}

function animateParticles(particles) {
    const points = particles.userData.points;
    const particlePositions = particles.userData.particlePositions;
    const particleProgress = particles.userData.particleProgress;

    function updateParticles() {
        const positionAttribute = particles.geometry.getAttribute('position');
        const particleCount = particleProgress.length;

        for (let i = 0; i < particleCount; i++) {
            particleProgress[i] += 0.01;
            particleProgress[i] = 0;

            const pointIndex = Math.floor(particleProgress[i] * (points.length - 1));
            const nextPointIndex = Math.min(pointIndex + 1, points.length - 1);
            const pointFraction = (particleProgress[i] * (points.length - 1)) - pointIndex;

            const currentPoint = points[pointIndex];
            const nextPoint = points[nextPointIndex];

            positionAttribute.setXYZ(
                i,
                currentPoint.x + (nextPoint.x - currentPoint.x) * pointFraction,
                currentPoint.y + (nextPoint.y - currentPoint.y) * pointFraction,
                currentPoint.z + (nextPoint.z - currentPoint.z) * pointFraction
            );
        }

        positionAttribute.needsUpdate = true;
        requestAnimationFrame(updateParticles);
    }

    updateParticles();
}

function updateStats() {
    document.querySelector('.stat-value:nth-child(1)').textContent = Object.keys(networkGraph.nodes).length;
    document.querySelector('.stat-value:nth-child(2)').textContent = networkGraph.edges.length;
    document.querySelector('.stat-value:nth-child(3)').textContent = honeypotPlacements.length;
    document.querySelector('.stat-value:nth-child(4)').textContent = '-- (Run optimization)';
}

function runOptimization(numHoneypots) {
    const algorithmType = document.getElementById('algorithm').value;
    const nodes = Object.keys(networkGraph.nodes);

    showLoading(true);

    honeypotPlacements = [];

    setTimeout(() => {
        switch (algorithmType) {
            case 'genetic':
                honeypotPlacements = geneticAlgorithmOptimization(nodes, numHoneypots);
                break;
            case 'simulated_annealing':
                honeypotPlacements = simulatedAnnealingOptimization(nodes, numHoneypots);
                break;
            case 'particle_swarm':
                honeypotPlacements = particleSwarmOptimization(nodes, numHoneypots);
                break;
            case 'ant_colony':
                honeypotPlacements = antColonyOptimization(nodes, numHoneypots);
                break;
            default:
                honeypotPlacements = geneticAlgorithmOptimization(nodes, numHoneypots);
        }

        updateHoneypotVisualization();

        const score = calculateOptimizationScore();
        document.querySelector('.stats-content .stat-item:nth-child(4) .stat-value').textContent = score.toFixed(2);

        showLoading(false);
    }, 2000);
}

function geneticAlgorithmOptimization(nodes, numHoneypots) {
    const highValueNodes = nodes.filter(node => networkGraph.nodes[node].asset_value > 5);
    const placements = [];

    for (let i = 0; i < Math.min(numHoneypots, highValueNodes.length); i++) {
        placements.push(highValueNodes[i]);
    }

    while (placements.length < numHoneypots) {
        let bestNode = null;
        let maxConnections = -1;

        for (const nodeId of nodes) {
            if (placements.includes(nodeId)) continue;

            const connections = networkGraph.edges.filter(edge =>
                edge.source === nodeId || edge.target === nodeId
            ).length;

            if (connections > maxConnections) {
                maxConnections = connections;
                bestNode = nodeId;
            }
        }

        if (bestNode) {
            placements.push(bestNode);
        } else {
            break;
        }
    }

    animateSelectionProcess(placements);

    return placements;
}

function simulatedAnnealingOptimization(nodes, numHoneypots) {
    const placements = [];
    const highRiskNodes = [...nodes].sort((a, b) =>
        (networkGraph.nodes[b].asset_value * networkGraph.nodes[b].vulnerability) -
        (networkGraph.nodes[a].asset_value * networkGraph.nodes[a].vulnerability)
    );

    for (let i = 0; i < Math.min(numHoneypots, highRiskNodes.length); i++) {
        placements.push(highRiskNodes[i]);
    }

    animateSelectionProcess(placements);
    return placements;
}

function particleSwarmOptimization(nodes, numHoneypots) {
    const coveredNodes = new Set();
    const placements = [];

    for (const path of attackPaths) {
        if (placements.length >= numHoneypots) break;

        const midIndex = Math.floor(path.length / 2);
        const midNode = path[midIndex];

        if (!coveredNodes.has(midNode)) {
            placements.push(midNode);
            coveredNodes.add(midNode);

            networkGraph.edges.forEach(edge => {
                if (edge.source === midNode) coveredNodes.add(edge.target);
                if (edge.target === midNode) coveredNodes.add(edge.source);
            });
        }
    }

    const remainingNodes = nodes.filter(node => !coveredNodes.has(node))
        .sort((a, b) => networkGraph.nodes[b].asset_value - networkGraph.nodes[a].asset_value);

    for (const node of remainingNodes) {
        if (placements.length >= numHoneypots) break;
        placements.push(node);
    }

    animateSelectionProcess(placements);
    return placements;
}

function antColonyOptimization(nodes, numHoneypots) {
    const placements = [];
    const nodeCentrality = {};

    nodes.forEach(node => {
        let pathsThrough = 0;

        attackPaths.forEach(path => {
            if (path.includes(node)) pathsThrough++;
        });

        nodeCentrality[node] = pathsThrough;
    });

    const sortedNodes = [...nodes].sort((a, b) => nodeCentrality[b] - nodeCentrality[a]);

    for (let i = 0; i < Math.min(numHoneypots, sortedNodes.length); i++) {
        placements.push(sortedNodes[i]);
    }

    animateSelectionProcess(placements);
    return placements;
}

function updateHoneypotVisualization() {
    Object.keys(nodeObjects).forEach(nodeId => {
        const nodeMesh = nodeObjects[nodeId];
        const assetValue = networkGraph.nodes[nodeId].asset_value;
        const assetValueNormalized = assetValue / 10;
        const nodeColor = new THREE.Color().setHSL(0.05, 0.8, 0.4 + assetValueNormalized * 0.4);

        nodeMesh.material.color.set(nodeColor);
        nodeMesh.material.emissive.copy(nodeColor).multiplyScalar(0.2);
        nodeMesh.scale.set(1, 1, 1);

        nodeMesh.children[0].material.color.copy(nodeColor).multiplyScalar(0.5);
        nodeMesh.children[0].material.opacity = 0.3;
        nodeMesh.children[0].scale.set(1, 1, 1);
    });

    honeypotPlacements.forEach(nodeId => {
        const nodeMesh = nodeObjects[nodeId];
        if (nodeMesh) {
            const honeypotColor = new THREE.Color(0x4fc3f7);
            nodeMesh.material.color.set(honeypotColor);
            nodeMesh.material.emissive.copy(honeypotColor).multiplyScalar(0.5);

            nodeMesh.scale.set(1.3, 1.3, 1.3);

            nodeMesh.children[0].material.color.set(honeypotColor);
            nodeMesh.children[0].material.opacity = 0.6;
            nodeMesh.children[0].scale.set(1.8, 1.8, 1.8);

            animateHoneypotGlow(nodeMesh);
        }
    });

    recalculateAttackPaths();
}

function animateSelectionProcess(placements) {
    const animationDelay = 300;

    placements.forEach((nodeId, index) => {
        setTimeout(() => {
            const nodeMesh = nodeObjects[nodeId];
            if (!nodeMesh) return;

            const originalScale = nodeMesh.scale.clone();
            const originalColor = nodeMesh.material.color.clone();

            gsap.to(nodeMesh.scale, {
                x: 2,
                y: 2,
                z: 2,
                duration: 0.2,
                ease: "power2.out",
                onComplete: () => {
                    gsap.to(nodeMesh.scale, {
                        x: originalScale.x,
                        y: originalScale.y,
                        z: originalScale.z,
                        duration: 0.2,
                        ease: "power2.in"
                    });
                }
            });

            gsap.to(nodeMesh.material.color, {
                r: 1,
                g: 1,
                b: 1,
                duration: 0.1,
                ease: "power1.out",
                onComplete: () => {
                    gsap.to(nodeMesh.material.color, {
                        r: originalColor.r,
                        g: originalColor.g,
                        b: originalColor.b,
                        duration: 0.3,
                        ease: "power1.in"
                    });
                }
            });
        }, index * animationDelay);
    });
}

function animateHoneypotGlow(nodeMesh) {
    const glowMesh = nodeMesh.children[0];
    const minScale = 1.5;
    const maxScale = 1.8;
    const minOpacity = 0.4;
    const maxOpacity = 0.7;

    gsap.to(glowMesh.scale, {
        x: maxScale,
        y: maxScale,
        z: maxScale,
        duration: 1.5,
        ease: "sine.inOut",
        repeat: -1,
        yoyo: true
    });

    gsap.to(glowMesh.material, {
        opacity: maxOpacity,
        duration: 1.5,
        ease: "sine.inOut",
        repeat: -1,
        yoyo: true
    });
}

function recalculateAttackPaths() {
    pathObjects.forEach(path => {
        scene.remove(path);
    });
    pathObjects = [];

    if (showAttackPaths) {
        createAttackPathVisuals();
    }
}

function calculateOptimizationScore() {
    if (honeypotPlacements.length === 0) return 0;

    let score = 0;
    const maxScore = 100;
    const honeypotCoverage = new Set();

    honeypotPlacements.forEach(nodeId => {
        networkGraph.edges.forEach(edge => {
            if (edge.source === nodeId) honeypotCoverage.add(edge.target);
            if (edge.target === nodeId) honeypotCoverage.add(edge.source);
        });

        honeypotCoverage.add(nodeId);
    });

    let pathsDisrupted = 0;
    attackPaths.forEach(path => {
        for (const nodeId of path) {
            if (honeypotPlacements.includes(nodeId)) {
                pathsDisrupted++;
                break;
            }
        }
    });

    const highValueAssets = Object.keys(networkGraph.nodes).filter(nodeId =>
        networkGraph.nodes[nodeId].asset_value > 5
    );

    let protectedAssets = 0;
    highValueAssets.forEach(nodeId => {
        if (honeypotCoverage.has(nodeId)) protectedAssets++;
    });

    const coverageScore = (honeypotCoverage.size / Object.keys(networkGraph.nodes).length) * 40;
    const pathScore = (pathsDisrupted / attackPaths.length) * 30;
    const assetScore = (protectedAssets / Math.max(1, highValueAssets.length)) * 30;

    score = coverageScore + pathScore + assetScore;
    return Math.min(score, maxScore);
}

function showLoading(show) {
    const loading = document.getElementById('loading');
    if (show) {
        loading.classList.add('active');
    } else {
        loading.classList.remove('active');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    initScene();

    document.getElementById('node-data').addEventListener('change', (e) => {
        const filename = e.target.files[0]?.name || 'No file selected';
        document.getElementById('node-filename').textContent = filename;
    });

    document.getElementById('edge-data').addEventListener('change', (e) => {
        const filename = e.target.files[0]?.name || 'No file selected';
        document.getElementById('edge-filename').textContent = filename;
    });

    document.getElementById('algorithm').addEventListener('change', (e) => {
        const algorithm = e.target.value;
        document.getElementById('algorithm-info').innerHTML = `<strong>${algorithm.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())} Algorithm:</strong> ${algorithmDescriptions[algorithm]}`;
    });

    document.getElementById('optimization-form').addEventListener('submit', async (e) => {
        e.preventDefault();

        const nodeFile = document.getElementById('node-data').files[0];
        const edgeFile = document.getElementById('edge-data').files[0];
        const numHoneypots = parseInt(document.getElementById('honeypots').value);

        if (!nodeFile || !edgeFile) {
            const alert = document.getElementById('alert');
            alert.textContent = 'Please upload both node and edge data files';
            alert.style.display = 'block';
            setTimeout(() => {
                alert.style.display = 'none';
            }, 5000);
            return;
        }

        try {
            showLoading(true);

            const nodeData = await parseCSV(nodeFile);
            const edgeData = await parseCSV(edgeFile);

            buildNetworkGraph(nodeData, edgeData);

            runOptimization(numHoneypots);
        } catch (error) {
            console.error('Error processing files:', error);
            const alert = document.getElementById('alert');
            alert.textContent = 'Error processing files. Please check your CSV format.';
            alert.style.display = 'block';
            showLoading(false);
        }
    });

    document.getElementById('rotate-toggle').addEventListener('click', () => {
        isRotating = !isRotating;
        document.getElementById('rotate-toggle').textContent = isRotating ? 'Pause Rotation' : 'Resume Rotation';
    });

    document.getElementById('attack-paths-toggle').addEventListener('click', () => {
        showAttackPaths = !showAttackPaths;
        document.getElementById('attack-paths-toggle').textContent = showAttackPaths ? 'Hide Attack Paths' : 'Show Attack Paths';

        if (showAttackPaths) {
            createAttackPathVisuals();
        } else {
            pathObjects.forEach(path => {
                scene.remove(path);
            });
            pathObjects = [];
        }
    });

    document.getElementById('reset-view').addEventListener('click', () => {
        camera.position.set(0, 0, 15);
        camera.lookAt(0, 0, 0);
        controls.reset();
    });

    if (!nodeFile && !edgeFile) {
        generateSampleData();
    }
});

function generateSampleData() {
    const nodeData = [];
    const edgeData = [];

    for (let i = 1; i <= 15; i++) {
        nodeData.push({
            id: `node${i}`,
            asset_value: Math.random() * 10,
            vulnerability: Math.random()
        });
    }

    for (let i = 1; i <= 15; i++) {
        const numConnections = 2 + Math.floor(Math.random() * 2);

        for (let j = 0; j < numConnections; j++) {
            let target;
            do {
                target = `node${1 + Math.floor(Math.random() * 15)}`;
            } while (target === `node${i}`);

            edgeData.push({
                source: `node${i}`,
                target: target,
                vulnerability: Math.random()
            });
        }
    }

    const nodeBlob = new Blob([
        'id,asset_value,vulnerability\n' +
        nodeData.map(node => `${node.id},${node.asset_value.toFixed(2)},${node.vulnerability.toFixed(2)}`).join('\n')
    ], { type: 'text/csv' });

    const edgeBlob = new Blob([
        'source,target,vulnerability\n' +
        edgeData.map(edge => `${edge.source},${edge.target},${edge.vulnerability.toFixed(2)}`).join('\n')
    ], { type: 'text/csv' });

    const nodeFile = new File([nodeBlob], 'sample_nodes.csv', { type: 'text/csv' });
    const edgeFile = new File([edgeBlob], 'sample_edges.csv', { type: 'text/csv' });

    const nodeDataTransfer = new DataTransfer();
    nodeDataTransfer.items.add(nodeFile);
    document.getElementById('node-data').files = nodeDataTransfer.files;
    document.getElementById('node-filename').textContent = 'sample_nodes.csv';

    const edgeDataTransfer = new DataTransfer();
    edgeDataTransfer.items.add(edgeFile);
    document.getElementById('edge-data').files = edgeDataTransfer.files;
    document.getElementById('edge-filename').textContent = 'sample_edges.csv';

    parseCSV(nodeFile).then(nodeData => {
        parseCSV(edgeFile).then(edgeData => {
            buildNetworkGraph(nodeData, edgeData);
        });
    });
}

function createNodeTooltip(nodeId, event) {
    const tooltip = document.createElement('div');
    tooltip.className = 'tooltip';
    document.body.appendChild(tooltip);

    const node = networkGraph.nodes[nodeId];
    tooltip.innerHTML = `
        <div><strong>Node:</strong> ${nodeId}</div>
        <div><strong>Asset Value:</strong> ${node.asset_value.toFixed(2)}</div>
        <div><strong>Vulnerability:</strong> ${(node.vulnerability * 100).toFixed(1)}%</div>
        ${honeypotPlacements.includes(nodeId) ? '<div><strong>Status:</strong> Honeypot</div>' : ''}
    `;

    tooltip.style.left = `${event.clientX}px`;
    tooltip.style.top = `${event.clientY - 10}px`;
    tooltip.style.opacity = '1';

    return tooltip;
}

renderer.domElement.addEventListener('mousemove', (event) => {
    const mouse = new THREE.Vector2();
    mouse.x = (event.clientX / renderer.domElement.clientWidth) * 2 - 1;
    mouse.y = -(event.clientY / renderer.domElement.clientHeight) * 2 + 1;

    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(mouse, camera);

    const intersects = raycaster.intersectObjects(Object.values(nodeObjects));

    const existingTooltip = document.querySelector('.tooltip');
    if (existingTooltip) {
        document.body.removeChild(existingTooltip);
    }

    if (intersects.length > 0) {
        const nodeObject = intersects[0].object;
        const nodeId = nodeObject.userData.id;

        if (nodeId) {
            createNodeTooltip(nodeId, event);
        }
    }
});

const form = document.getElementById('optimization-form');
const sampleDataBtn = document.createElement('button');
sampleDataBtn.type = 'button';
sampleDataBtn.textContent = 'Load Sample Data';
sampleDataBtn.style.marginTop = '20px';
sampleDataBtn.style.background = '#253652';
sampleDataBtn.style.color = '#d9e3f1';

sampleDataBtn.addEventListener('click', () => {
    generateSampleData();
});

form.appendChild(sampleDataBtn);