export class NetworkVisualizer {
    constructor(canvasId, options = {}) {
        // Default configuration
        this.config = {
            nodeSize: options.nodeSize || 1.5,
            nodeSpacing: options.nodeSpacing || 5, // Constant spacing between nodes
            defaultColor: options.defaultColor ? options.defaultColor : new BABYLON.Color3(0.7, 0.2, 0.2),
            highlightColor: options.highlightColor ? options.highlightColor : new BABYLON.Color3(0.3, 0.76, 0.97),
            edgeColor: options.edgeColor ? options.edgeColor : new BABYLON.Color3(0.3, 0.5, 0.7),
            edgeHighlightColor: options.edgeHighlightColor ? options.edgeHighlightColor : new BABYLON.Color3(1, 0.67, 0.25),
            backgroundColor: options.backgroundColor ? options.backgroundColor : new BABYLON.Color4(0.05, 0.05, 0.1, 1),
            initialCameraRadius: options.initialCameraRadius || 25,
            showLabels: options.showLabels !== undefined ? options.showLabels : true,
            autoRotate: options.autoRotate !== undefined ? options.autoRotate : true
        };

        // Core properties
        this.canvasId = canvasId;
        this.engine = null;
        this.scene = null;
        this.camera = null;
        this.nodeMeshes = {};
        this.edgeLines = [];
        this.systemsData = [];
        this.selectedNode = null;
        this.highlightMaterial = null;
        this.isRotating = this.config.autoRotate;
        this.tooltipElement = null;
        this.lastPointerPosition = { x: 0, y: 0 };  // Track the last pointer position

        // Initialize the scene
        this.initScene();
        this.createTooltip();
    }

    // Initialize Babylon.js scene
    initScene() {
        const canvas = document.getElementById(this.canvasId);
        if (!canvas) {
            console.error(`Canvas element with ID '${this.canvasId}' not found.`);
            return;
        }

        this.engine = new BABYLON.Engine(canvas, true);
        this.scene = new BABYLON.Scene(this.engine);
        this.scene.clearColor = this.config.backgroundColor;

        // Setup camera
        this.camera = new BABYLON.ArcRotateCamera(
            "camera", 
            Math.PI / 2, 
            Math.PI / 3, 
            this.config.initialCameraRadius, 
            BABYLON.Vector3.Zero(), 
            this.scene
        );
        this.camera.attachControl(canvas, true);
        this.camera.lowerRadiusLimit = 5;
        this.camera.upperRadiusLimit = 50;

        // Add lights
        const hemisphericLight = new BABYLON.HemisphericLight(
            "hemisphericLight", 
            new BABYLON.Vector3(1, 1, 0), 
            this.scene
        );
        hemisphericLight.intensity = 0.7;

        const pointLight = new BABYLON.PointLight(
            "pointLight", 
            new BABYLON.Vector3(0, 0, 0), 
            this.scene
        );
        pointLight.intensity = 0.5;

        // Create highlight material
        this.highlightMaterial = new BABYLON.StandardMaterial("highlightMaterial", this.scene);
        this.highlightMaterial.diffuseColor = this.config.highlightColor;
        this.highlightMaterial.specularColor = new BABYLON.Color3(1, 1, 1);
        this.highlightMaterial.specularPower = 32;
        this.highlightMaterial.emissiveColor = new BABYLON.Color3(0.1, 0.3, 0.4);

        // Track pointer position for tooltip
        this.scene.onPointerMove = (evt) => {
            this.lastPointerPosition = {
                x: evt.clientX || evt.pointerX,
                y: evt.clientY || evt.pointerY
            };
        };

        // Start rendering loop
        this.engine.runRenderLoop(() => {
            if (this.isRotating) {
                this.camera.alpha += 0.005;
            }
            this.scene.render();
        });

        // Handle window resize
        window.addEventListener("resize", () => {
            this.engine.resize();
        });
    }

    // Create tooltip for hover details
    createTooltip() {
        // Remove existing tooltip if any
        const existingTooltip = document.getElementById('node-tooltip');
        if (existingTooltip) {
            existingTooltip.remove();
        }

        this.tooltipElement = document.createElement('div');
        this.tooltipElement.id = 'node-tooltip';
        this.tooltipElement.style.position = 'absolute';
        this.tooltipElement.style.display = 'none';
        this.tooltipElement.style.background = 'rgba(16, 18, 27, 0.9)';
        this.tooltipElement.style.color = '#e6e6e6';
        this.tooltipElement.style.padding = '10px 15px';
        this.tooltipElement.style.borderRadius = '8px';
        this.tooltipElement.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.5)';
        this.tooltipElement.style.backdropFilter = 'blur(10px)';
        this.tooltipElement.style.border = '1px solid rgba(255, 255, 255, 0.1)';
        this.tooltipElement.style.zIndex = '100';
        this.tooltipElement.style.maxWidth = '250px';
        this.tooltipElement.style.fontSize = '14px';
        this.tooltipElement.style.pointerEvents = 'none'; // Prevent interference with mouse events

        document.body.appendChild(this.tooltipElement);
    }

    // Update tooltip content and position
    updateTooltip(evt, node) {
        const system = this.systemsData.find(s => s.ip === node.ip);
        if (!system) return;

        // Set tooltip content
        this.tooltipElement.innerHTML = `
            <div style="font-weight: bold; margin-bottom: 5px; color: #4fc3f7;">${system.name}</div>
            <div style="margin-bottom: 5px;"><span style="opacity: 0.7;">IP:</span> ${system.ip}</div>
            <div><span style="opacity: 0.7;">Connections:</span> ${system.connections.length}</div>
        `;

        // Use last known pointer position if event doesn't have coordinates
        const x = this.lastPointerPosition.x || 0;
        const y = this.lastPointerPosition.y || 0;

        // Position tooltip near cursor
        this.tooltipElement.style.left = (x + 15) + 'px';
        this.tooltipElement.style.top = (y + 15) + 'px';

        // Show tooltip
        this.tooltipElement.style.display = 'block';
    }

    // Hide tooltip
    hideTooltip() {
        this.tooltipElement.style.display = 'none';
    }

    // Parse CSV data
    parseCSV(content) {
        const systems = [];
        const lines = content.trim().split('\n');
        
        lines.forEach(line => {
            if (!line.trim()) return;
            const [name, ip, connections] = line.split(',');
            systems.push({
                name: name.trim(),
                ip: ip.trim(),
                connections: connections ? connections.split('|').map(c => c.trim()) : []
            });
        });
        
        return systems;
    }

    // Load data from string
    loadDataFromString(csvString) {
        const systems = this.parseCSV(csvString);
        this.visualizeNetwork(systems);
        return systems;
    }

    // Load data from file
    loadDataFromFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = (e) => {
                try {
                    const systems = this.parseCSV(e.target.result);
                    this.visualizeNetwork(systems);
                    resolve(systems);
                } catch (error) {
                    reject(error);
                }
            };
            
            reader.onerror = (e) => {
                reject(new Error('Error reading file'));
            };
            
            reader.readAsText(file);
        });
    }

    // Clear the scene
    clearScene() {
        // Dispose of all node meshes
        for (const nodeId in this.nodeMeshes) {
            this.nodeMeshes[nodeId].dispose();
        }
        
        // Dispose of all edge lines
        this.edgeLines.forEach(line => {
            line.dispose();
        });
        
        this.nodeMeshes = {};
        this.edgeLines = [];
        this.selectedNode = null;
    }

    // Create node label with Babylon GUI
    createNodeLabel(name, ip, nodeMesh) {
        if (!this.config.showLabels) return null;

        const plane = BABYLON.MeshBuilder.CreatePlane("label-" + ip, { size: 3 }, this.scene);
        plane.parent = nodeMesh;
        plane.position.y = 2.2; // Position above the node
        plane.billboardMode = BABYLON.Mesh.BILLBOARDMODE_ALL;
    
        const advancedTexture = BABYLON.GUI.AdvancedDynamicTexture.CreateForMesh(plane);
    
        const textBlock = new BABYLON.GUI.TextBlock();
        textBlock.text = name;
        textBlock.color = "rgba(253, 250, 250, 0.4)";
        textBlock.fontSize = 200;
        textBlock.outlineWidth = 20;
        textBlock.outlineColor = "black";
        textBlock.background = "rgba(255, 255, 255, 0.4)";
        textBlock.textHorizontalAlignment = BABYLON.GUI.Control.HORIZONTAL_ALIGNMENT_CENTER;
        textBlock.textVerticalAlignment = BABYLON.GUI.Control.VERTICAL_ALIGNMENT_CENTER;
    
        advancedTexture.addControl(textBlock);
    
        return plane;
    }

    // Create glow effect around node
    createGlowLayer(nodeMesh, color) {
        const glowMaterial = new BABYLON.StandardMaterial("glowMaterial", this.scene);
        glowMaterial.emissiveColor = color;
        glowMaterial.alpha = 0.5;
        
        const glowSphere = BABYLON.MeshBuilder.CreateSphere(
            "glow", 
            { diameter: this.config.nodeSize * 2 }, 
            this.scene
        );
        glowSphere.material = glowMaterial;
        glowSphere.parent = nodeMesh;
        
        return glowSphere;
    }

    // Force-directed layout algorithm to position nodes with constant spacing
    calculateNodePositions(systems) {
        const nodeCount = systems.length;
        const positions = {};
        
        // Create a map of node connections for the algorithm
        const connections = {};
        systems.forEach(system => {
            connections[system.ip] = system.connections;
        });
        
        // Initial positions - distribute nodes in a sphere
        systems.forEach((system, index) => {
            // Calculate points on a sphere for initial placement
            const phi = Math.acos(-1 + (2 * index) / nodeCount);
            const theta = Math.sqrt(nodeCount * Math.PI) * phi;
            
            const x = 15 * Math.sin(phi) * Math.cos(theta);
            const y = 15 * Math.sin(phi) * Math.sin(theta);
            const z = 15 * Math.cos(phi);
            
            positions[system.ip] = new BABYLON.Vector3(x, y, z);
        });
        
        // Apply force-directed algorithm to maintain spacing
        const iterations = 100;
        const k = this.config.nodeSpacing; // Optimal distance between nodes (constant spacing)
        
        for (let iter = 0; iter < iterations; iter++) {
            const forces = {};
            systems.forEach(system => {
                forces[system.ip] = new BABYLON.Vector3(0, 0, 0);
            });
            
            // Calculate repulsive forces (every node repels every other node)
            for (let i = 0; i < systems.length; i++) {
                const nodeA = systems[i];
                const posA = positions[nodeA.ip];
                
                for (let j = i + 1; j < systems.length; j++) {
                    const nodeB = systems[j];
                    const posB = positions[nodeB.ip];
                    
                    // Calculate direction and distance
                    const direction = posA.subtract(posB);
                    const distance = direction.length();
                    
                    if (distance === 0) continue; // Skip if nodes are at the same position
                    
                    // Normalize direction
                    direction.normalize();
                    
                    // Calculate repulsive force (inverse square law)
                    const repulsiveForce = k * k / distance;
                    
                    // Apply force to both nodes in opposite directions
                    forces[nodeA.ip].addInPlace(direction.scale(repulsiveForce));
                    forces[nodeB.ip].subtractInPlace(direction.scale(repulsiveForce));
                }
            }
            
            // Calculate attractive forces (connected nodes attract each other)
            systems.forEach(system => {
                system.connections.forEach(targetIp => {
                    if (!positions[targetIp]) return;
                    
                    const posA = positions[system.ip];
                    const posB = positions[targetIp];
                    
                    // Calculate direction and distance
                    const direction = posB.subtract(posA);
                    const distance = direction.length();
                    
                    if (distance === 0) return; // Skip if nodes are at the same position
                    
                    // Normalize direction
                    direction.normalize();
                    
                    // Calculate attractive force (linear)
                    const attractiveForce = distance * distance / k;
                    
                    // Apply force
                    forces[system.ip].addInPlace(direction.scale(attractiveForce));
                });
            });
            
            // Apply forces to update positions
            systems.forEach(system => {
                const damping = 0.1; // Damping factor to prevent oscillations
                positions[system.ip].addInPlace(forces[system.ip].scale(damping));
            });
        }
        
        return positions;
    }

    // Visualize the network
    visualizeNetwork(systems) {
        this.clearScene();
        this.systemsData = systems;
        
        // Calculate node positions with constant spacing
        const nodePositions = this.calculateNodePositions(systems);
        
        // Create nodes
        systems.forEach(system => {
            const position = nodePositions[system.ip];
            
            // Create node sphere
            const sphere = BABYLON.MeshBuilder.CreateSphere(
                `node-${system.ip}`, 
                { diameter: this.config.nodeSize }, 
                this.scene
            );
            sphere.position = position;
            
            // Create material
            const nodeMaterial = new BABYLON.StandardMaterial(`material-${system.ip}`, this.scene);
            nodeMaterial.diffuseColor = this.config.defaultColor;
            nodeMaterial.specularColor = new BABYLON.Color3(1, 1, 1);
            nodeMaterial.specularPower = 32;
            
            sphere.material = nodeMaterial;
            sphere.ip = system.ip;
            sphere.name = system.name;
            
            // Create node label if enabled
            if (this.config.showLabels) {
                this.createNodeLabel(system.name, system.ip, sphere);
            }
            
            // Create glow effect
            this.createGlowLayer(sphere, this.config.defaultColor);
            
            // Store node mesh
            this.nodeMeshes[system.ip] = sphere;
            
            // Add action manager for interactivity
            sphere.actionManager = new BABYLON.ActionManager(this.scene);
            
            // Hover effects
            sphere.actionManager.registerAction(
                new BABYLON.ExecuteCodeAction(
                    BABYLON.ActionManager.OnPointerOverTrigger,
                    (evt) => {
                        document.body.style.cursor = 'pointer';
                        // For tooltip, use the last known pointer position
                        this.updateTooltip(null, sphere);
                        
                        // Highlight node on hover if not selected
                        if (this.selectedNode !== sphere) {
                            sphere.scaling = new BABYLON.Vector3(1.3, 1.3, 1.3);
                        }
                    }
                )
            );
            
            sphere.actionManager.registerAction(
                new BABYLON.ExecuteCodeAction(
                    BABYLON.ActionManager.OnPointerOutTrigger,
                    (evt) => {
                        document.body.style.cursor = 'default';
                        this.hideTooltip();
                        
                        // Return to normal size if not selected
                        if (this.selectedNode !== sphere) {
                            sphere.scaling = new BABYLON.Vector3(1, 1, 1);
                        }
                    }
                )
            );
            
            // Click action
            sphere.actionManager.registerAction(
                new BABYLON.ExecuteCodeAction(
                    BABYLON.ActionManager.OnPickTrigger,
                    (evt) => {
                        this.selectNode(sphere);
                    }
                )
            );
        });
        
        // Create edges/connections with slight curve
        systems.forEach(system => {
            const sourceNode = this.nodeMeshes[system.ip];
            
            if (!sourceNode) return;
            
            system.connections.forEach(targetIp => {
                const targetNode = this.nodeMeshes[targetIp];
                
                if (!targetNode) return;
                
                // Check if this connection already exists in the reverse direction
                const edgeExists = this.edgeLines.some(line => 
                    line.name === `edge-${targetIp}-${system.ip}`
                );
                
                if (edgeExists) return;
                
                const sourcePos = sourceNode.position;
                const targetPos = targetNode.position;
                
                // Create a curved path
                const midPoint = new BABYLON.Vector3(
                    (sourcePos.x + targetPos.x) / 2,
                    (sourcePos.y + targetPos.y) / 2,
                    (sourcePos.z + targetPos.z) / 2 + 1 // Add a small offset for the curve
                );
                
                // Create points for the curve
                const curvePoints = [];
                for (let t = 0; t <= 1; t += 0.05) {
                    // Quadratic Bezier curve
                    const point = new BABYLON.Vector3(
                        (1-t)*(1-t)*sourcePos.x + 2*(1-t)*t*midPoint.x + t*t*targetPos.x,
                        (1-t)*(1-t)*sourcePos.y + 2*(1-t)*t*midPoint.y + t*t*targetPos.y,
                        (1-t)*(1-t)*sourcePos.z + 2*(1-t)*t*midPoint.z + t*t*targetPos.z
                    );
                    curvePoints.push(point);
                }
                
                const line = BABYLON.MeshBuilder.CreateLines(
                    `edge-${system.ip}-${targetIp}`, 
                    { points: curvePoints }, 
                    this.scene
                );
                line.color = this.config.edgeColor;
                
                this.edgeLines.push(line);
            });
        });
        
        // Trigger event callbacks
        if (this.onNetworkVisualized) {
            this.onNetworkVisualized({
                nodes: systems.length,
                edges: this.countTotalEdges(),
                mostConnectedNode: this.findMostConnectedNode()
            });
        }
    }

    // Select a node
    selectNode(node) {
        // Deselect previously selected node
        if (this.selectedNode) {
            const prevMaterial = new BABYLON.StandardMaterial(`material-${this.selectedNode.ip}`, this.scene);
            prevMaterial.diffuseColor = this.config.defaultColor;
            prevMaterial.specularColor = new BABYLON.Color3(1, 1, 1);
            prevMaterial.specularPower = 32;
            
            this.selectedNode.material = prevMaterial;
            this.selectedNode.scaling = new BABYLON.Vector3(1, 1, 1);
            
            // Restore connections color
            this.highlightNodeConnections(this.selectedNode.ip, false);
        }
        
        // If clicking the same node, just deselect
        if (this.selectedNode === node) {
            this.selectedNode = null;
            
            if (this.onNodeDeselected) {
                this.onNodeDeselected();
            }
            
            return;
        }
        
        // Select new node
        this.selectedNode = node;
        this.selectedNode.material = this.highlightMaterial;
        this.selectedNode.scaling = new BABYLON.Vector3(1.3, 1.3, 1.3);
        
        // Highlight connections
        this.highlightNodeConnections(this.selectedNode.ip, true);
        
        // Trigger callback
        if (this.onNodeSelected) {
            const system = this.systemsData.find(s => s.ip === node.ip);
            if (system) {
                this.onNodeSelected(system);
            }
        }
    }

    // Highlight node connections
    highlightNodeConnections(nodeIp, highlight) {
        this.edgeLines.forEach(line => {
            const [source, target] = line.name.replace('edge-', '').split('-');
            if (source === nodeIp || target === nodeIp) {
                line.color = highlight ? 
                    this.config.edgeHighlightColor : 
                    this.config.edgeColor;
                
                // Make highlighted lines thicker
                line.width = highlight ? 3 : 1;
            }
        });
    }

    // Count total edges in the network
    countTotalEdges() {
        let count = 0;
        this.systemsData.forEach(system => {
            count += system.connections.length;
        });
        // Each connection is counted twice (once from each end)
        return count / 2;
    }

    // Find most connected node
    findMostConnectedNode() {
        if (this.systemsData.length === 0) return null;
        
        let mostConnections = 0;
        let mostConnectedNode = null;
        
        this.systemsData.forEach(system => {
            if (system.connections.length > mostConnections) {
                mostConnections = system.connections.length;
                mostConnectedNode = system;
            }
        });
        
        return mostConnectedNode;
    }

    // Toggle rotation
    toggleRotation() {
        this.isRotating = !this.isRotating;
        return this.isRotating;
    }

    // Reset camera view
    resetCamera() {
        this.camera.alpha = Math.PI / 2;
        this.camera.beta = Math.PI / 3;
        this.camera.radius = this.config.initialCameraRadius;
    }

    // Update node size
    updateNodeSize(size) {
        this.config.nodeSize = size;
        
        if (Object.keys(this.nodeMeshes).length > 0) {
            const scaleFactor = size / 1.5;
            for (const nodeId in this.nodeMeshes) {
                const nodeMesh = this.nodeMeshes[nodeId];
                
                // Apply scaling only if it's not the selected node
                if (this.selectedNode !== nodeMesh) {
                    nodeMesh.scaling = new BABYLON.Vector3(scaleFactor, scaleFactor, scaleFactor);
                } else {
                    // For selected node, make it slightly larger
                    nodeMesh.scaling = new BABYLON.Vector3(scaleFactor * 1.3, scaleFactor * 1.3, scaleFactor * 1.3);
                }
            }
        }
        
        return this.config.nodeSize;
    }

    // Camera zoom controls
    zoomIn(amount = 2) {
        if (this.camera.radius > this.camera.lowerRadiusLimit) {
            this.camera.radius -= amount;
        }
        return this.camera.radius;
    }
    
    zoomOut(amount = 2) {
        if (this.camera.radius < this.camera.upperRadiusLimit) {
            this.camera.radius += amount;
        }
        return this.camera.radius;
    }
    
    resetZoom() {
        this.camera.radius = this.config.initialCameraRadius;
        return this.camera.radius;
    }

    // Generate sample data for testing
    generateSampleData() {
        const sampleData = 
        `SystemA,192.168.0.1,192.168.0.2|192.168.0.3
        SystemB,192.168.0.2,192.168.0.1|192.168.0.4
        SystemC,192.168.0.3,192.168.0.1
        SystemD,192.168.0.4,192.168.0.2|192.168.0.5
        SystemE,192.168.0.5,192.168.0.4|192.168.0.6|192.168.0.7
        SystemF,192.168.0.6,192.168.0.5
        SystemG,192.168.0.7,192.168.0.5|192.168.0.8
        SystemH,192.168.0.8,192.168.0.7|192.168.0.9
        SystemI,192.168.0.9,192.168.0.8|192.168.0.10
        SystemJ,192.168.0.10,192.168.0.9|192.168.0.1`;
    
        return this.parseCSV(sampleData);
    }
    
    // Load sample data directly
    loadSampleData() {
        const systems = this.generateSampleData();
        this.visualizeNetwork(systems);
        return systems;
    }
    
    // Event callbacks - to be set by user
    onNodeSelected = null;
    onNodeDeselected = null;
    onNetworkVisualized = null;
    
    // Dispose of all resources
    dispose() {
        // Clear scene
        this.clearScene();
        
        // Remove tooltip
        if (this.tooltipElement) {
            this.tooltipElement.remove();
            this.tooltipElement = null;
        }
        
        // Stop render loop
        this.engine.stopRenderLoop();
        
        // Dispose of scene
        this.scene.dispose();
        
        // Dispose of engine
        this.engine.dispose();
    }
}