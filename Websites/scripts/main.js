 // Global variables
 let engine,scene,camera;
 let nodeMeshes={};
 let edgeLines=[];
 let isRotating=true;
 let selectedNode=null;
 let highlightMaterial;
 let systemsData=[];
 let nodeSize=1.5;
 let file = null;
 const API_URL='http://localhost:8000';

 function initScene() {
     const canvas=document.getElementById('canvas');
     engine=new BABYLON.Engine(canvas,true);
     
     scene=new BABYLON.Scene(engine);
     scene.clearColor=new BABYLON.Color4(0.05,0.05,0.1,1);
     
     camera=new BABYLON.ArcRotateCamera("camera",Math.PI/2,Math.PI/3,25,BABYLON.Vector3.Zero(),scene);
     camera.attachControl(canvas,true);
     camera.lowerRadiusLimit=5;
     camera.upperRadiusLimit=50;
     
     // Add lights
     const light1=new BABYLON.HemisphericLight("light1",new BABYLON.Vector3(1,1,0),scene);
     light1.intensity=0.7;
     
     const light2=new BABYLON.PointLight("light2",new BABYLON.Vector3(0,0,0),scene);
     light2.intensity=0.5;
     
     // Create highlight material
     highlightMaterial=new BABYLON.StandardMaterial("highlightMaterial",scene);
     highlightMaterial.diffuseColor=new BABYLON.Color3(0.3,0.76,0.97);
     highlightMaterial.specularColor=new BABYLON.Color3(1,1,1);
     highlightMaterial.specularPower=32;
     highlightMaterial.emissiveColor=new BABYLON.Color3(0.1,0.3,0.4);
     
     // Start rendering
     engine.runRenderLoop(() => {
         if (isRotating) {
             camera.alpha += 0.005;
        }
         scene.render();
    });
     
     // Handle window resize
     window.addEventListener("resize",() => {
         engine.resize();
    });
}
 
 // Parse CSV data
 function parseCSV(content) {
     const systems=[];
     const lines=content.trim().split('\n');
     
     lines.forEach(line => {
         if (!line.trim()) return;
         const [name,ip,connections]=line.split(',');
         systems.push({
             name: name.trim(),
             ip: ip.trim(),
             connections: connections ? connections.split('|').map(c => c.trim()) : []
        });
    });
     
     return systems;
}
 
 // Create node label with Babylon GUI
 function createNodeLabel(name,ip,nodeMesh) {
     const plane=BABYLON.MeshBuilder.CreatePlane("label-" + ip,{size: 3},scene);
     plane.parent=nodeMesh;
     plane.position.y=2.2;
     plane.billboardMode=BABYLON.Mesh.BILLBOARDMODE_ALL;
 
     const advancedTexture=BABYLON.GUI.AdvancedDynamicTexture.CreateForMesh(plane);
 
     const textBlock=new BABYLON.GUI.TextBlock();
     textBlock.text=name;
     textBlock.color="rgb(255,255,255)";
     textBlock.fontSize=200;
     textBlock.outlineWidth=20;
     textBlock.outlineColor="black";
     textBlock.background="rgba(255,255,255,0.4)"
     textBlock.textHorizontalAlignment=BABYLON.GUI.Control.HORIZONTAL_ALIGNMENT_CENTER;
     textBlock.textVerticalAlignment=BABYLON.GUI.Control.VERTICAL_ALIGNMENT_CENTER;
 
     advancedTexture.addControl(textBlock);
 
     return plane;
}
 
 
 // Create glow effect around node
 function createGlowLayer(nodeMesh,color) {
     const glowMaterial=new BABYLON.StandardMaterial("glowMaterial",scene);
     glowMaterial.emissiveColor=color;
     glowMaterial.alpha=0.5;
     
     const glowSphere=BABYLON.MeshBuilder.CreateSphere("glow",{diameter: nodeSize * 2},scene);
     glowSphere.material=glowMaterial;
     glowSphere.parent=nodeMesh;
     
     return glowSphere;
}
 
 // Clear the scene
 function clearScene() {
     if (scene) {
         for (const nodeId in nodeMeshes) {
             nodeMeshes[nodeId].dispose();
        }
         
         edgeLines.forEach(line => {
             line.dispose();
        });
         
         nodeMeshes={};
         edgeLines=[];
    }
}
 
 // Show loading indicator
 function showLoading(show) {
     const loading=document.getElementById('loading');
     if (loading) {
         if (show) {
             loading.classList.add('active');
        } else {
             loading.classList.remove('active');
        }
    }
}
 
 function updateNetworkStats() {
     document.getElementById('total-nodes').textContent=systemsData.length;
     document.getElementById('total-edges').textContent=countTotalEdges();
     document.getElementById('most-connected').textContent=findMostConnectedNode();
}

 function countTotalEdges() {
     let count=0;
     systemsData.forEach(system => {
         count += system.connections.length;
    });
    return count;
}
 
 function findMostConnectedNode() {
     if (systemsData.length === 0) return "--";
     
     let mostConnections=0;
     let mostConnectedNode=null;
     
     systemsData.forEach(system => {
         if (system.connections.length > mostConnections) {
             mostConnections=system.connections.length;
             mostConnectedNode=system.name;
        }
    });
     
     return mostConnectedNode;
}
 
 // Show node inspector panel
 function showNodeInspector(node) {
     if (!node) return;
     
     const system=systemsData.find(s => s.ip === node.ip);
     if (!system) return;
     
     document.getElementById('node-name').textContent=system.name;
     document.getElementById('node-ip').textContent=system.ip;
     document.getElementById('node-connections-count').textContent=system.connections.length;
     
     const connectionsList=document.getElementById('connections-list');
     connectionsList.innerHTML='';
     
     system.connections.forEach(conn => {
         const connectedSystem=systemsData.find(s => s.ip === conn);
         if (connectedSystem) {
             const item=document.createElement('div');
             item.className='connection-item';
             item.textContent=`${connectedSystem.name} (${conn})`;
             connectionsList.appendChild(item);
        }
    });
     
     document.getElementById('inspector').classList.add('visible');
}
 
 // Hide node inspector panel
 function hideNodeInspector() {
     document.getElementById('inspector').classList.remove('visible');
}
 
 // Highlight all connections for a node
 function highlightNodeConnections(nodeIp,highlight) {
     edgeLines.forEach(line => {
         const [source,target]=line.name.replace('edge-','').split('-');
         if (source === nodeIp || target === nodeIp) {
             line.color=highlight ? 
                new BABYLON.Color3(0.3,0.5,0.7):
                 new BABYLON.Color3(1,0.67,0.25); 
             if (highlight) {
                 line.width=3;
            } else {
                 line.width=1;
            }
        }
    });
}
 
 // Build network visualization
 function buildNetworkVisualization(systems) {
     showLoading(true);
     clearScene();
     systemsData=systems;
     // Tooltip hover events


     setTimeout(() => {
         // Create nodes
        const baseRadius=10;      
        const scalingFactor=0.2;  
        const numNodes=systems.length;
        const radius=baseRadius + scalingFactor * numNodes;
        const zRange=Math.max(5,radius * 0.2);
        systems.forEach((system,index) => {
            const angle=(index/numNodes) * Math.PI * 2;
            const x=radius * Math.cos(angle);
            const y=radius * Math.sin(angle);
            const z=(Math.random() - 0.5) * zRange;
            const sphere=BABYLON.MeshBuilder.CreateSphere(`node-${system.ip}`,{diameter: nodeSize},scene);
            sphere.position=new BABYLON.Vector3(x,y,z);
             sphere.position=new BABYLON.Vector3(x,y,z);
             sphere.actionManager=new BABYLON.ActionManager(scene);
             sphere.actionManager.registerAction(
                 new BABYLON.ExecuteCodeAction(
                     BABYLON.ActionManager.OnPointerOverTrigger,
                     function(evt) {
                         updateTooltip(evt.pointerEvent,sphere);
                    }
                 )
             );
     
             sphere.actionManager.registerAction(
                 new BABYLON.ExecuteCodeAction(
                     BABYLON.ActionManager.OnPointerOutTrigger,
                     function(evt) {
                         hideTooltip();
                    }
                 )
             );
             // Create material
             const nodeMaterial=new BABYLON.StandardMaterial(`material-${system.ip}`,scene);
             const nodeColor=new BABYLON.Color3(0.7,0.2,0.2); // Red color
             nodeMaterial.diffuseColor=nodeColor;
             nodeMaterial.specularColor=new BABYLON.Color3(1,1,1);
             nodeMaterial.specularPower=32;
             
             sphere.material=nodeMaterial;
             sphere.ip=system.ip;
             sphere.name=system.name;
             
             // Create node label
             const label=createNodeLabel(system.name,system.ip,sphere);
             
             // Create glow effect
             const glow=createGlowLayer(sphere,nodeColor);
             
             // Store node mesh
             nodeMeshes[system.ip]=sphere;
             
             // Add action manager for interactivity
             sphere.actionManager=new BABYLON.ActionManager(scene);
             
             // Hover effects
             sphere.actionManager.registerAction(
                 new BABYLON.ExecuteCodeAction(
                     BABYLON.ActionManager.OnPointerOverTrigger,
                     function(evt) {
                         document.body.style.cursor='pointer';
                         // Highlight node on hover if not selected
                         if (selectedNode !== sphere) {
                             sphere.scaling=new BABYLON.Vector3(1.3,1.3,1.3);
                        }
                    }
                 )
             );
             
             sphere.actionManager.registerAction(
                 new BABYLON.ExecuteCodeAction(
                     BABYLON.ActionManager.OnPointerOutTrigger,
                     function(evt) {
                         document.body.style.cursor='default';
                         // Return to normal size if not selected
                         if (selectedNode !== sphere) {
                             sphere.scaling=new BABYLON.Vector3(1,1,1);
                        }
                    }
                 )
             );
             
             // Click action
             sphere.actionManager.registerAction(
                 new BABYLON.ExecuteCodeAction(
                     BABYLON.ActionManager.OnPickTrigger,
                     function(evt) {
                         selectNode(sphere);
                    }
                 )
             );
        });
         
         // Create connections/edges with slight curve
         systems.forEach(system => {
             const sourceNode=nodeMeshes[system.ip];
             
             if (!sourceNode) return;
             
             system.connections.forEach(targetIp => {
                 const targetNode=nodeMeshes[targetIp];
                 
                 if (!targetNode) return;
                 
                 // Check if this connection already exists in the reverse direction
                 const edgeExists=edgeLines.some(line => 
                     line.name === `edge-${targetIp}-${system.ip}`
                 );
                 
                 if (edgeExists) return;
                 
                 const sourcePos=sourceNode.position;
                 const targetPos=targetNode.position;
                 
                 // Create a curved path
                 const midPoint=new BABYLON.Vector3(
                     (sourcePos.x + targetPos.x)/2,
                     (sourcePos.y + targetPos.y)/2,
                     (sourcePos.z + targetPos.z)/2 + 1 
                 );
                 
                 // Create points for the curve
                 const curvePoints=[];
                 for (let t=0; t <= 1; t += 0.1) {
                     // Quadratic Bezier curve
                     const point=new BABYLON.Vector3(
                         (1-t)*(1-t)*sourcePos.x + 2*(1-t)*t*midPoint.x + t*t*targetPos.x,
                         (1-t)*(1-t)*sourcePos.y + 2*(1-t)*t*midPoint.y + t*t*targetPos.y,
                         (1-t)*(1-t)*sourcePos.z + 2*(1-t)*t*midPoint.z + t*t*targetPos.z
                     );
                     curvePoints.push(point);
                }
                 
                 const line=BABYLON.MeshBuilder.CreateLines(`edge-${system.ip}-${targetIp}`,{points: curvePoints},scene);
                 line.color=new BABYLON.Color3(0.3,0.5,0.7); // Blue color
                 
                 edgeLines.push(line);
            });
        });
         
         updateNetworkStats();
         showLoading(false);
    },500);
}
 
 // Select a node
 function selectNode(node) {
     // Deselect previously selected node
     if (selectedNode) {
         const prevMaterial=new BABYLON.StandardMaterial(`material-${selectedNode.ip}`,scene);
         prevMaterial.diffuseColor=new BABYLON.Color3(0.7,0.2,0.2);
         prevMaterial.specularColor=new BABYLON.Color3(1,1,1);
         prevMaterial.specularPower=32;
         
         selectedNode.material=prevMaterial;
         selectedNode.scaling=new BABYLON.Vector3(1,1,1);
         
         // Restore connections color
         highlightNodeConnections(selectedNode.ip,false);
    }
     
     // If clicking the same node,just deselect
     if (selectedNode === node) {
         selectedNode=null;
         hideNodeInspector();
         return;
    }
     
     // Select new node
     selectedNode=node;
     selectedNode.material=highlightMaterial;
     selectedNode.scaling=new BABYLON.Vector3(1.3,1.3,1.3);
     
     // Highlight connections
     highlightNodeConnections(selectedNode.ip,true);
     
     // Show node details
     showNodeInspector(node);
}
 function generateSampleData() {
     const sampleData=
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
 
     return parseCSV(sampleData);
}

 // Create tooltip for hover details
function createTooltip() {
const tooltipContainer=document.createElement('div');
tooltipContainer.id='node-tooltip';
tooltipContainer.style.position='absolute';
tooltipContainer.style.display='none';
tooltipContainer.style.background='rgba(16,18,27,0.9)';
tooltipContainer.style.color='#e6e6e6';
tooltipContainer.style.padding='10px 15px';
tooltipContainer.style.borderRadius='8px';
tooltipContainer.style.boxShadow='0 4px 20px rgba(0,0,0,0.5)';
tooltipContainer.style.backdropFilter='blur(10px)';
tooltipContainer.style.border='1px solid rgba(255,255,255,0.1)';
tooltipContainer.style.zIndex='100';
tooltipContainer.style.maxWidth='250px';
tooltipContainer.style.fontSize='14px';
tooltipContainer.style.pointerEvents='none'; 

document.body.appendChild(tooltipContainer);
return tooltipContainer;
}

// Update tooltip content and position
function updateTooltip(event,node) {
const system=systemsData.find(s => s.ip === node.ip);
if (!system) return;

const tooltip=document.getElementById('node-tooltip');
if (!tooltip) return;

// Set tooltip content
tooltip.innerHTML=`
 <div style="font-weight: bold; margin-bottom: 5px; color: #4fc3f7;">${system.name}</div>
 <div style="margin-bottom: 5px;"><span style="opacity: 0.7;">IP:</span> ${system.ip}</div>
 <div><span style="opacity: 0.7;">Connections:</span> ${system.connections.length}</div>
`;

// Position tooltip near cursor
tooltip.style.left=(event.clientX + 15) + 'px';
tooltip.style.top=(event.clientY + 15) + 'px';

// Show tooltip
tooltip.style.display='block';
}

// Hide tooltip
function hideTooltip() {
const tooltip=document.getElementById('node-tooltip');
if (tooltip) {
 tooltip.style.display='none';
}
}
async function handlePredictClick() {
    const resultDiv=document.getElementById('prediction-result');
    resultDiv.innerHTML=`<div class="loading-text">Predicting honeypots...</div>`;
    const predictButton=document.getElementById('predict-btn');
    predictButton.disabled=true;

    if (!systemsData || systemsData.length === 0) {
        resultDiv.innerHTML=`<div class="error-text">No network data loaded.</div>`;
        predictButton.disabled=false;
        return;
}

    // --- Prepare richer network data for the backend ---
    const networkDataForApi=systemsData.map(node => ({
       name: node.name,
       ip: node.ip,
       connections: Array.isArray(node.connections) ? node.connections : [],
       asset_value: 1.0 + (node.connections?.length || 0) * 0.5,// Example derivation
       os_type: "Ubuntu 20.04",// Default or from CSV/UI
       open_ports: [22,80,443]   // Default or from CSV/UI
}));

    // --- Get Algorithm Choice and Parameters ---
    // **** MODIFIED: Read value from the dropdown ****
    const algorithmSelect=document.getElementById('algorithm-select');
    const selectedAlgorithm=algorithmSelect.value; // Get the selected value ("GA","WOA",or "HHO")

    // TODO: Add UI element (e.g.,slider or input) for max_honeypots if needed
    const maxHoneypots=3; // Using default for now

    // --- Construct the Request Payload ---
    const requestPayload={
       network_data: networkDataForApi,
       algorithm_type: selectedAlgorithm,// Use the value read from the dropdown
       max_honeypots: maxHoneypots
       // Optional: Add specific algorithm params like iterations if you add UI for them
};

    console.log("Sending prediction request:",JSON.stringify(requestPayload,null,2));

    try {
        // --- Send the Fetch Request (no change here) ---
        const response=await fetch(`${API_URL}/predict`,{
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
        },
            body: JSON.stringify(requestPayload)
    });

        // --- Handle Potential Errors (no change here) ---
        if (!response.ok) {
            let errorMsg=`Error: ${response.status} ${response.statusText}`;
            try {
                const errorData=await response.json();
                errorMsg=`Error ${response.status}: ${errorData.detail || JSON.stringify(errorData)}`;
        } catch (e) {/* Ignore if response body isn't JSON */}
            throw new Error(errorMsg);
    }

        // --- Process the Response (no change here) ---
        const result=await response.json();
        console.log("Received prediction:",result);

        // --- Display the Enhanced Results (no change here) ---
        let resultHTML=`
            <span>Algorithm: <strong>${result.algorithm_used || 'N/A'}</strong></span>
            <span>${result.message || 'Prediction complete.'}</span>
            ${result.fitness_score !== null && result.fitness_score !== undefined ? `<span>Fitness: <strong>${result.fitness_score.toFixed(4)}</strong></span>` : ''}
        `;

        if (result.predicted_honeypots && result.predicted_honeypots.length > 0) {
            const predictedNodesInfo=result.predicted_honeypots.map(ip => {
                const node=systemsData.find(n => n.ip === ip);
                return node ? `${node.name} (${ip})` : ip;
        });
            resultHTML += '<ul>';
            predictedNodesInfo.forEach(nodeInfo => {
                resultHTML += `<li>${nodeInfo}</li>`;
        });
            resultHTML += '</ul>';
    } else {
            resultHTML += '<i>No specific honeypots recommended.</i>';
    }
        resultDiv.innerHTML=resultHTML;

} catch (error) {
        console.error('Prediction failed:',error);
        resultDiv.innerHTML=`<div class="error-text">Prediction failed: ${error.message}. Check backend logs.</div>`;
} finally {
         predictButton.disabled=false;
}
}
 // Initialize the application
 function init() {
     // Initialize Babylon.js scene
     initScene();
     createTooltip();
     // Set up event listeners
     document.getElementById('csv-data').addEventListener('input',handleFileUpload);
     document.getElementById('sample-data').addEventListener('click',loadSampleData);
     document.getElementById('rotate-toggle').addEventListener('click',toggleRotation);
     document.getElementById('reset-view').addEventListener('click',resetCamera);
     document.getElementById('close-inspector').addEventListener('click',hideNodeInspector);
     document.getElementById('node-size').addEventListener('input',updateNodeSize);
     document.getElementById('zoom-in').addEventListener('click',zoomIn);
     document.getElementById('zoom-out').addEventListener('click',zoomOut);
     document.getElementById('reset-zoom').addEventListener('click',resetZoom);
     document.getElementById('predict-btn').addEventListener('click',handlePredictClick);
}
 
 // Handle file upload
 function handleFileUpload(event) {
    file=event.target.files[0];
     
     if (!file) return;
     
     document.getElementById('filename').textContent=file.name;
     
     const reader=new FileReader();
     
     reader.onload=function(e) {
         const systems=parseCSV(e.target.result);
         buildNetworkVisualization(systems);
    };
     
     reader.readAsText(file);
}
 
 // Load sample data
 function loadSampleData() {
     const systems=generateSampleData();
     file = null;
     buildNetworkVisualization(systems);
}
 
 // Toggle rotation
 function toggleRotation() {
     isRotating=!isRotating;
     document.getElementById('rotate-toggle').textContent=isRotating ? 'Pause Rotation' : 'Resume Rotation';
}
 
 // Reset camera view
 function resetCamera() {
     camera.alpha=Math.PI/2;
     camera.beta=Math.PI/3;
     camera.radius=25;
}
 
 // Update node size
 function updateNodeSize(event) {
     nodeSize=parseFloat(event.target.value);
     document.getElementById('size-value').textContent=nodeSize.toFixed(1);
     
     if (Object.keys(nodeMeshes).length > 0) {
         for (const nodeId in nodeMeshes) {
             const nodeMesh=nodeMeshes[nodeId];
             nodeMesh.scaling=new BABYLON.Vector3(nodeSize/1.5,nodeSize/1.5,nodeSize/1.5);
        }
    }
}
 
 // Camera zoom controls
 function zoomIn() {
     if (camera.radius > camera.lowerRadiusLimit) {
         camera.radius -= 2;
    }
}
 
 function zoomOut() {
     if (camera.radius < camera.upperRadiusLimit) {
         camera.radius += 2;
    }
}
 
 function resetZoom() {
     camera.radius=25;
}
 
 // Initialize on window load
 window.addEventListener('load',init);