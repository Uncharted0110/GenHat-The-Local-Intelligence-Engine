/**
 * Mindmap Visualization Component
 * Creates a new window with particle background and interactive React Flow mindmap
 */

// Type definition for mindmap tree structure
export interface MindmapNode {
  id: string
  label: string
  collapsed: boolean
  children: MindmapNode[]
}

/**
 * Show interactive mindmap visualization in a new window
 * @param mindmapData - Hierarchical tree data structure for the mindmap
 * @param title - Title to display in the mindmap window
 */
export function showMindmapVisualization(mindmapData: MindmapNode, title: string = 'Mind Map') {
  const windowWidth = 1400
  const windowHeight = 900
  const screenX = (window.screen.width - windowWidth) / 2
  const screenY = (window.screen.height - windowHeight) / 2

  // Create the mindmap HTML with React Flow
  const mindmapHTML = generateMindmapHTML(mindmapData, title)

  const blob = new Blob([mindmapHTML], { type: 'text/html' })
  const url = URL.createObjectURL(blob)
  const windowFeatures = `width=${windowWidth},height=${windowHeight},left=${screenX},top=${screenY}`
  window.open(url, 'mindmap', windowFeatures)

  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

/**
 * Legacy function for showing markdown content (kept for backwards compatibility)
 */
export function showMindmapMarkdown(mindmapContent: string, title: string = 'Mind Map') {
  const windowWidth = 1200
  const windowHeight = 800
  const screenX = (window.screen.width - windowWidth) / 2
  const screenY = (window.screen.height - windowHeight) / 2

  const mindmapHTML = `
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>${title}</title>
      <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif; overflow: hidden; height: 100vh; width: 100vw; }
        #canvas { position: fixed; left: 0; top: 0; width: 100%; height: 100%; background-color: #000000; z-index: 1; }
        #mindmap-content { position: fixed; left: 0; top: 0; width: 100%; height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; z-index: 2; padding: 40px; overflow-y: auto; }
        .mindmap-container { background: rgba(13, 13, 13, 0.85); border: 2px solid #ff8c00; border-radius: 16px; padding: 32px; max-width: 900px; width: 100%; box-shadow: 0 8px 32px rgba(255, 140, 0, 0.3); backdrop-filter: blur(10px); }
        .mindmap-title { color: #ff8c00; font-size: 28px; font-weight: 700; margin-bottom: 24px; text-align: center; display: flex; align-items: center; justify-content: center; gap: 12px; }
        .mindmap-content { color: #e0e0e0; line-height: 1.8; font-size: 15px; }
        .mindmap-content h1, .mindmap-content h2, .mindmap-content h3 { color: #ff8c00; margin: 20px 0 12px 0; }
        .mindmap-content h1 { font-size: 24px; }
        .mindmap-content h2 { font-size: 20px; }
        .mindmap-content h3 { font-size: 16px; }
        .mindmap-content strong { color: #ffa500; }
        .mindmap-content ul, .mindmap-content ol { margin: 12px 0 12px 24px; }
        .mindmap-content li { margin-bottom: 8px; }
        .mindmap-content code { background: #1a1a1a; padding: 2px 6px; border-radius: 4px; color: #ff8c00; font-family: monospace; }
        .mindmap-content pre { background: #1a1a1a; padding: 12px; border-radius: 6px; overflow-x: auto; margin: 12px 0; border-left: 3px solid #ff8c00; }
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #ff8c00; border-radius: 4px; }
      </style>
    </head>
    <body>
      <canvas id="canvas"></canvas>
      <div id="mindmap-content">
        <div class="mindmap-container">
          <div class="mindmap-title">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="5" r="3"></circle>
              <circle cx="6" cy="12" r="3"></circle>
              <circle cx="18" cy="12" r="3"></circle>
              <circle cx="9" cy="20" r="3"></circle>
              <circle cx="15" cy="20" r="3"></circle>
              <line x1="12" y1="8" x2="6" y2="9"></line>
              <line x1="12" y1="8" x2="18" y2="9"></line>
              <line x1="6" y1="15" x2="9" y2="17"></line>
              <line x1="18" y1="15" x2="15" y2="17"></line>
            </svg>
            ${title}
          </div>
          <div class="mindmap-content" id="mindmap-text">${mindmapContent}</div>
        </div>
      </div>
      <script>
        let canvas = document.querySelector("#canvas");
        let ctx = canvas.getContext("2d");
        let w, h, particles;
        let particleDistance = 40;
        let mouse = { x: undefined, y: undefined, radius: 100 };
        function init() { resizeReset(); animationLoop(); }
        function resizeReset() {
          w = canvas.width = window.innerWidth;
          h = canvas.height = window.innerHeight;
          particles = [];
          for (let y = (((h - particleDistance) % particleDistance) + particleDistance) / 2; y < h; y += particleDistance) {
            for (let x = (((w - particleDistance) % particleDistance) + particleDistance) / 2; x < w; x += particleDistance) {
              particles.push(new Particle(x, y));
            }
          }
        }
        function animationLoop() { ctx.clearRect(0, 0, w, h); drawScene(); requestAnimationFrame(animationLoop); }
        function drawScene() { for (let i = 0; i < particles.length; i++) { particles[i].update(); particles[i].draw(); } }
        function mousemove(e) { mouse.x = e.x; mouse.y = e.y; }
        function mouseout() { mouse.x = undefined; mouse.y = undefined; }
        class Particle {
          constructor(x, y) { this.x = x; this.y = y; this.size = 2; this.baseX = this.x; this.baseY = this.y; this.speed = (Math.random() * 25) + 5; }
          draw() { ctx.fillStyle = "rgba(255, 140, 0, 0.5)"; ctx.beginPath(); ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2); ctx.closePath(); ctx.fill(); }
          update() {
            let dx = mouse.x - this.x; let dy = mouse.y - this.y; let distance = Math.sqrt(dx * dx + dy * dy);
            let force = (mouse.radius - distance) / mouse.radius;
            let directionX = dx / distance; let directionY = dy / distance;
            if (distance < mouse.radius) { this.x -= directionX * force * this.speed; this.y -= directionY * force * this.speed; }
            else { if (this.x !== this.baseX) { this.x -= (this.x - this.baseX) / 10; } if (this.y !== this.baseY) { this.y -= (this.y - this.baseY) / 10; } }
          }
        }
        init();
        window.addEventListener("resize", resizeReset);
        window.addEventListener("mousemove", mousemove);
        window.addEventListener("mouseout", mouseout);
      </script>
    </body>
    </html>
  `

  const blob = new Blob([mindmapHTML], { type: 'text/html' })
  const url = URL.createObjectURL(blob)
  const windowFeatures = `width=${windowWidth},height=${windowHeight},left=${screenX},top=${screenY}`
  window.open(url, 'mindmap', windowFeatures)

  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

/**
 * Generate the full HTML for the React Flow mindmap
 */
function generateMindmapHTML(treeData: MindmapNode, title: string): string {
  // Fallback for undefined treeData
  if (!treeData) {
    console.error('generateMindmapHTML called with undefined treeData')
    treeData = {
      id: 'error',
      label: 'Error: No data',
      collapsed: false,
      children: []
    }
  }
  
  const titleBase = String(title || 'Mind Map')
  const escapedTitle = (titleBase as any).replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;')
  const treeJson = JSON.stringify(treeData)
  const treeDataJson = (treeJson as any).replaceAll('<', String.raw`\u003c`).replaceAll('>', String.raw`\u003e`)

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${escapedTitle}</title>
  <style>
    /* Minimal React Flow base styles (inlined to satisfy CSP) */
    .react-flow { position: relative; width: 100%; height: 100%; overflow: hidden; }
    .react-flow__renderer { position: absolute; left: 0; top: 0; right: 0; bottom: 0; outline: none; }
    .react-flow__pane { position: absolute; left: 0; top: 0; right: 0; bottom: 0; cursor: grab; }
    .react-flow__pane.dragging { cursor: grabbing; }
    .react-flow__selection { position: absolute; pointer-events: none; }
    .react-flow__node { position: absolute; user-select: none; transform-origin: center center; }
    .react-flow__handle { position: absolute; border-radius: 100%; cursor: crosshair; }
    .react-flow__edge { pointer-events: none; }
    .react-flow__edge-path { fill: none; }
    .react-flow__container { width: 100%; height: 100%; }
    .react-flow__edges { position: absolute; left: 0; top: 0; right: 0; bottom: 0; overflow: visible; }
    .react-flow__nodes { position: absolute; left: 0; top: 0; right: 0; bottom: 0; overflow: visible; }
    .react-flow__controls { display: flex; flex-direction: column; }
    .react-flow__controls-button { appearance: none; cursor: pointer; }
    .react-flow__background { position: absolute; left:0; top:0; right:0; bottom:0; }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif; overflow: hidden; height: 100vh; width: 100vw; background: #0d0d0d; }
    #particle-canvas { position: fixed; left: 0; top: 0; width: 100%; height: 100%; z-index: 1; pointer-events: none; }
    #mindmap-root { position: fixed; left: 0; top: 0; width: 100%; height: 100%; z-index: 2; }
    
    /* React Flow overrides */
    .react-flow__renderer { background: transparent !important; }
    .react-flow__background { background-color: transparent !important; }
    .react-flow__controls { background: rgba(26, 26, 26, 0.95) !important; border: 1px solid #ff8c00 !important; border-radius: 8px !important; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4) !important; }
    .react-flow__controls-button { background: transparent !important; border-bottom: 1px solid rgba(255, 140, 0, 0.3) !important; color: #ff8c00 !important; fill: #ff8c00 !important; width: 28px !important; height: 28px !important; }
    .react-flow__controls-button:last-child { border-bottom: none !important; }
    .react-flow__controls-button:hover { background: rgba(255, 140, 0, 0.2) !important; }
    .react-flow__controls-button svg { fill: #ff8c00 !important; max-width: 14px !important; max-height: 14px !important; }
    .react-flow__attribution { display: none !important; }
    
    /* Edge styles */
    .react-flow__edge-path { stroke: #ff8c00 !important; stroke-width: 2px !important; }
    .react-flow__edge.animated path { stroke-dasharray: 5 !important; animation: dashdraw 0.5s linear infinite !important; }
    @keyframes dashdraw { from { stroke-dashoffset: 10; } }
    
    /* Node styles */
    .mindmap-node {
      display: flex;
      align-items: center;
      padding: 10px 14px;
      background: rgba(26, 26, 26, 0.95);
      border: 2px solid #555;
      border-radius: 8px;
      min-width: 120px;
      max-width: 220px;
      min-height: 40px;
      transition: all 0.2s ease;
      backdrop-filter: blur(8px);
    }
    .mindmap-node:hover { border-color: #ff8c00; box-shadow: 0 4px 16px rgba(255, 140, 0, 0.3); }
    .mindmap-node.selected { border-color: #00bfff !important; box-shadow: 0 0 0 3px rgba(0, 191, 255, 0.5), 0 4px 20px rgba(0, 191, 255, 0.4) !important; background: rgba(0, 191, 255, 0.1) !important; }
    .mindmap-node.root { background: linear-gradient(135deg, rgba(255, 140, 0, 0.2) 0%, rgba(255, 107, 0, 0.15) 100%); border-color: #ff8c00; }
    .mindmap-node.root.selected { background: linear-gradient(135deg, rgba(0, 191, 255, 0.2) 0%, rgba(0, 150, 200, 0.15) 100%) !important; border-color: #00bfff !important; }
    
    .mindmap-node .node-content { flex: 1; min-width: 0; }
    .mindmap-node .node-label { color: #e0e0e0; font-size: 13px; font-weight: 500; word-wrap: break-word; display: block; }
    .mindmap-node.root .node-label { color: #ff8c00; font-size: 15px; font-weight: 600; }
    .mindmap-node .node-input { background: transparent; border: 1px solid #ff8c00; border-radius: 4px; padding: 2px 4px; outline: none; color: #e0e0e0; font-size: 13px; font-weight: 500; width: 100%; font-family: inherit; }
    
    .mindmap-node .node-actions { display: flex; align-items: center; gap: 2px; margin-left: 8px; opacity: 0; transition: opacity 0.2s ease; }
    .mindmap-node:hover .node-actions { opacity: 1; }
    .mindmap-node .action-btn { display: flex; align-items: center; justify-content: center; width: 22px; height: 22px; background: rgba(255, 140, 0, 0.1); border: none; border-radius: 4px; color: #888; cursor: pointer; transition: all 0.15s ease; }
    .mindmap-node .action-btn:hover { background: rgba(255, 140, 0, 0.3); color: #ff8c00; }
    
    .mindmap-node .collapse-indicator { display: flex; align-items: center; justify-content: center; width: 20px; height: 20px; margin-left: 6px; background: rgba(255, 140, 0, 0.15); border: 1px solid #ff8c00; border-radius: 50%; color: #ff8c00; cursor: pointer; font-size: 11px; font-weight: bold; transition: all 0.15s ease; }
    .mindmap-node .collapse-indicator:hover { background: rgba(255, 140, 0, 0.3); transform: scale(1.1); }
    
    /* Handle styles - critical for edge alignment */
    .react-flow__handle { width: 8px !important; height: 8px !important; background: #ff8c00 !important; border: 2px solid #1a1a1a !important; }
    .react-flow__handle-left { left: -4px !important; }
    .react-flow__handle-right { right: -4px !important; }
    
    /* Header and controls */
    .mindmap-header { position: absolute; top: 16px; left: 16px; display: flex; align-items: center; gap: 12px; padding: 10px 16px; background: rgba(26, 26, 26, 0.95); border: 1px solid #ff8c00; border-radius: 10px; z-index: 10; backdrop-filter: blur(8px); }
    .mindmap-header svg { width: 22px; height: 22px; color: #ff8c00; stroke: #ff8c00; }
    .mindmap-header h1 { color: #ff8c00; font-size: 16px; font-weight: 600; margin: 0; }
    
    .control-panel { position: absolute; top: 16px; right: 16px; display: flex; gap: 8px; z-index: 10; }
    .panel-btn { display: flex; align-items: center; justify-content: center; gap: 6px; padding: 8px 14px; background: rgba(26, 26, 26, 0.95); border: 1px solid #ff8c00; border-radius: 8px; color: #ff8c00; font-size: 13px; font-weight: 500; cursor: pointer; transition: all 0.2s ease; backdrop-filter: blur(8px); }
    .panel-btn:hover { background: rgba(255, 140, 0, 0.15); }
    .panel-btn svg { width: 14px; height: 14px; stroke: currentColor; }
  </style>
</head>
<body>
  <canvas id="particle-canvas"></canvas>
  <div id="mindmap-root"></div>

  <script>
    // Particle background animation
    (function initParticles() {
      const canvas = document.getElementById('particle-canvas');
      const ctx = canvas.getContext('2d');
      let w, h, particles = [];
      const particleDistance = 40;
      const mouse = { x: undefined, y: undefined, radius: 100 };
      
      class Particle {
        constructor(x, y) {
          this.x = x; this.y = y; this.size = 2;
          this.baseX = x; this.baseY = y;
          this.speed = (Math.random() * 25) + 5;
        }
        draw() {
          ctx.fillStyle = "rgba(255, 140, 0, 0.4)";
          ctx.beginPath();
          ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
          ctx.closePath();
          ctx.fill();
        }
        update() {
          if (mouse.x === undefined || mouse.y === undefined) {
            if (this.x !== this.baseX) this.x -= (this.x - this.baseX) / 10;
            if (this.y !== this.baseY) this.y -= (this.y - this.baseY) / 10;
            return;
          }
          const dx = mouse.x - this.x;
          const dy = mouse.y - this.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          if (distance < mouse.radius) {
            const force = (mouse.radius - distance) / mouse.radius;
            const dirX = dx / distance;
            const dirY = dy / distance;
            this.x -= dirX * force * this.speed;
            this.y -= dirY * force * this.speed;
          } else {
            if (this.x !== this.baseX) this.x -= (this.x - this.baseX) / 10;
            if (this.y !== this.baseY) this.y -= (this.y - this.baseY) / 10;
          }
        }
      }
      
      function resizeReset() {
        w = canvas.width = window.innerWidth;
        h = canvas.height = window.innerHeight;
        particles = [];
        for (let y = particleDistance / 2; y < h; y += particleDistance) {
          for (let x = particleDistance / 2; x < w; x += particleDistance) {
            particles.push(new Particle(x, y));
          }
        }
      }
      
      function animate() {
        ctx.clearRect(0, 0, w, h);
        particles.forEach(p => { p.update(); p.draw(); });
        requestAnimationFrame(animate);
      }
      
      resizeReset();
      animate();
      window.addEventListener('resize', resizeReset);
      window.addEventListener('mousemove', e => { mouse.x = e.clientX; mouse.y = e.clientY; });
      window.addEventListener('mouseout', () => { mouse.x = undefined; mouse.y = undefined; });
    })();
  </script>

  <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
  <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
  <script src="https://unpkg.com/reactflow@11.10.1/dist/umd/index.js"></script>
  <script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>
  
  <script>
    (function() {
      const { useState, useCallback, useEffect, useRef } = React;
      const RF = window.ReactFlow;
      const { ReactFlow, Controls, Background, useNodesState, useEdgesState, ReactFlowProvider, useReactFlow, Handle, Position } = RF;

      function generateId() {
        return 'n' + Math.random().toString(36).slice(2, 9);
      }

      // Dagre layout function
      function getLayoutedElements(nodes, edges) {
        const g = new dagre.graphlib.Graph();
        g.setDefaultEdgeLabel(() => ({}));
        g.setGraph({ rankdir: 'LR', nodesep: 60, ranksep: 120, marginx: 50, marginy: 50 });

        nodes.forEach(node => {
          g.setNode(node.id, { width: 180, height: 50 });
        });

        edges.forEach(edge => {
          g.setEdge(edge.source, edge.target);
        });

        dagre.layout(g);

        const layoutedNodes = nodes.map(node => {
          const nodeWithPosition = g.node(node.id);
          return {
            ...node,
            position: {
              x: nodeWithPosition.x - 90,
              y: nodeWithPosition.y - 25
            }
          };
        });

        return { nodes: layoutedNodes, edges };
      }
      
      // Custom node component
      function MindMapNode({ data, id, selected }) {
        const [labelText, setLabelText] = useState(data.label);
        const inputRef = useRef(null);
        
        useEffect(() => { setLabelText(data.label); }, [data.label]);
        
        const handleDoubleClick = (e) => { e.stopPropagation(); };
        const finishEditing = () => { setIsEditing(false); if (labelText !== data.label && data.onLabelChange) data.onLabelChange(id, labelText); };
        const handleKeyDown = (e) => { if (e.key === 'Enter') finishEditing(); if (e.key === 'Escape') { setLabelText(data.label); setIsEditing(false); } };
        const handleAddChild = (e) => { e.stopPropagation(); if (data.onAddChild) data.onAddChild(id); };
        const handleDelete = (e) => { e.stopPropagation(); if (data.onDelete) data.onDelete(id); };
        const handleToggleCollapse = (e) => { e.stopPropagation(); if (data.onToggleCollapse) data.onToggleCollapse(id); };
        
        const isEditSelected = data.editMode && data.selected;
        const nodeClass = 'mindmap-node' + (data.isRoot ? ' root' : '') + (selected ? ' selected' : '') + (isEditSelected ? ' edit-selected' : '');
        const handleSelectClick = (e) => {
          if (!data.editMode) return;
          e.stopPropagation();
          data.onToggleSelect?.(id);
        };

        return React.createElement('div', { className: nodeClass, onClick: handleSelectClick },
          !data.isRoot && React.createElement(Handle, { type: 'target', position: Position.Left, id: 'target' }),
          React.createElement('div', { className: 'node-content' },
            React.createElement('span', { className: 'node-label', onDoubleClick: handleDoubleClick }, labelText)
          ),
          data.hasChildren && React.createElement('button', { className: 'collapse-indicator', type: 'button', onClick: handleToggleCollapse, title: data.collapsed ? 'Expand' : 'Collapse' }, data.collapsed ? '+' : '−'),
          React.createElement(Handle, { type: 'source', position: Position.Right, id: 'source' })
        );
      }
      
      const nodeTypes = { mindMap: MindMapNode };

      function normalizeTree(node, isRoot = true) {
        if (!node || typeof node !== 'object') {
          return { id: isRoot ? 'root' : generateId(), label: isRoot ? 'Mind Map' : 'Untitled', collapsed: false, children: [] };
        }
        return {
          id: node.id || generateId(),
          label: node.label || (isRoot ? 'Mind Map' : 'Untitled'),
          collapsed: typeof node.collapsed === 'boolean' ? node.collapsed : false,
          children: Array.isArray(node.children) ? node.children.map(child => normalizeTree(child, false)) : []
        };
      }

      // Convert tree to flat nodes/edges
      function treeToNodesEdges(tree, callbacks) {
        const nodes = [];
        const edges = [];
        
        function traverse(node, parentId) {
          const hasChildren = node.children && node.children.length > 0;
          nodes.push({
            id: node.id,
            type: 'mindMap',
            position: { x: 0, y: 0 },
            draggable: true,
            data: {
              label: node.label,
              isRoot: !parentId,
              collapsed: node.collapsed,
              hasChildren,
              onAddChild: callbacks.onAddChild,
              onToggleCollapse: callbacks.onToggleCollapse,
              onLabelChange: callbacks.onLabelChange,
              onDelete: callbacks.onDelete
            }
          });
          if (parentId) {
            edges.push({
              id: 'e-' + parentId + '-' + node.id,
              source: parentId,
              target: node.id,
              type: 'smoothstep',
              animated: true,
              style: { stroke: '#ff8c00', strokeWidth: 2 }
            });
          }
          if (!node.collapsed && node.children) {
            node.children.forEach(child => traverse(child, node.id));
          }
        }
        
        traverse(tree, null);
        return { nodes, edges };
      }
      
      // Main mindmap content
      function MindMapContent({ initialData, title }) {
        const [treeData, setTreeData] = useState(() => normalizeTree(initialData));
        const [nodes, setNodes, onNodesChange] = useNodesState([]);
        const [edges, setEdges, onEdgesChange] = useEdgesState([]);
        const { fitView, zoomIn, zoomOut } = useReactFlow();
        const initialLayoutDone = useRef(false);
        const [editMode, setEditMode] = useState(false);
        const [selectedIds, setSelectedIds] = useState(new Set());

        // Tree manipulation callbacks - must be stable
        const handleAddChild = useCallback((parentId) => {
          setTreeData(prev => {
            const newChild = { id: generateId(), label: 'New Node', collapsed: false, children: [] };
            const addTo = (node) => {
              if (node.id === parentId) return { ...node, collapsed: false, children: [...(node.children || []), newChild] };
              return { ...node, children: (node.children || []).map(addTo) };
            };
            return addTo(prev);
          });
        }, []);
        
        const handleToggleCollapse = useCallback((nodeId) => {
          setTreeData(prev => {
            const toggle = (node) => {
              if (node.id === nodeId) return { ...node, collapsed: !node.collapsed };
              return { ...node, children: (node.children || []).map(toggle) };
            };
            return toggle(prev);
          });
        }, []);
        
        const handleLabelChange = useCallback((nodeId, newLabel) => {
          setTreeData(prev => {
            const update = (node) => {
              if (node.id === nodeId) return { ...node, label: newLabel };
              return { ...node, children: (node.children || []).map(update) };
            };
            return update(prev);
          });
        }, []);
        
        const handleDelete = useCallback((nodeId) => {
          setTreeData(prev => {
            const remove = (node) => ({ ...node, children: (node.children || []).filter(c => c.id !== nodeId).map(remove) });
            return remove(prev);
          });
        }, []);

        // Build nodes/edges from tree and apply layout ONCE on tree structure change
        useEffect(() => {
          const callbacks = { onAddChild: handleAddChild, onToggleCollapse: handleToggleCollapse, onLabelChange: handleLabelChange, onDelete: handleDelete };
          let { nodes: rawNodes, edges: rawEdges } = treeToNodesEdges(treeData, callbacks);
          // inject editMode/selection handlers
          rawNodes = rawNodes.map(n => ({
            ...n,
            data: {
              ...n.data,
              editMode,
              selected: selectedIds.has(n.id),
              onToggleSelect: (id) => {
                setSelectedIds(prev => {
                  const next = new Set(prev);
                  if (next.has(id)) next.delete(id); else next.add(id);
                  return next;
                });
              }
            }
          }));
          const { nodes: layoutedNodes, edges: layoutedEdges } = getLayoutedElements(rawNodes, rawEdges);
          setNodes(layoutedNodes);
          setEdges(layoutedEdges);
          initialLayoutDone.current = true;
        }, [treeData, handleAddChild, handleToggleCollapse, handleLabelChange, handleDelete, setNodes, setEdges, editMode, selectedIds]);
        
        // Fit view after initial layout
        useEffect(() => {
          if (initialLayoutDone.current) {
            const timer = setTimeout(() => fitView({ padding: 0.2, duration: 300 }), 150);
            return () => clearTimeout(timer);
          }
        }, [nodes.length, fitView]);
        
        const handleFitView = () => fitView({ padding: 0.2, duration: 300 });
        const handleResetLayout = useCallback(() => {
          const callbacks = { onAddChild: handleAddChild, onToggleCollapse: handleToggleCollapse, onLabelChange: handleLabelChange, onDelete: handleDelete };
          const { nodes: rawNodes, edges: rawEdges } = treeToNodesEdges(treeData, callbacks);
          const { nodes: layoutedNodes, edges: layoutedEdges } = getLayoutedElements(rawNodes, rawEdges);
          setNodes(layoutedNodes);
          setEdges(layoutedEdges);
          setTimeout(() => fitView({ padding: 0.2, duration: 300 }), 100);
        }, [treeData, handleAddChild, handleToggleCollapse, handleLabelChange, handleDelete, setNodes, setEdges, fitView]);

        const toggleEditMode = () => {
          setEditMode(m => !m);
          if (editMode) {
            // clearing selection when turning off
            setSelectedIds(new Set());
          }
        };
        
        return React.createElement('div', { style: { width: '100%', height: '100%' } },
          React.createElement('div', { className: 'mindmap-header' },
            React.createElement('svg', { viewBox: '0 0 24 24', fill: 'none', strokeWidth: 2 },
              React.createElement('circle', { cx: 12, cy: 5, r: 3 }),
              React.createElement('circle', { cx: 6, cy: 12, r: 3 }),
              React.createElement('circle', { cx: 18, cy: 12, r: 3 }),
              React.createElement('line', { x1: 12, y1: 8, x2: 6, y2: 9 }),
              React.createElement('line', { x1: 12, y1: 8, x2: 18, y2: 9 })
            ),
            React.createElement('h1', null, title)
          ),
          React.createElement('div', { className: 'control-panel' },
            React.createElement('button', { className: 'panel-btn', onClick: () => zoomIn({ duration: 200 }), title: 'Zoom In' }, '+ Zoom'),
            React.createElement('button', { className: 'panel-btn', onClick: () => zoomOut({ duration: 200 }), title: 'Zoom Out' }, '− Zoom'),
            React.createElement('button', { className: 'panel-btn', onClick: handleFitView, title: 'Fit View' }, '⊡ Fit'),
            React.createElement('button', { className: 'panel-btn', onClick: handleResetLayout, title: 'Reset Layout' }, '↺ Reset'),
            React.createElement('button', { className: 'panel-btn', onClick: toggleEditMode, title: 'Toggle Edit Mode' }, editMode ? 'Edit: ON' : 'Edit: OFF')
          ),
          React.createElement(ReactFlow, {
            nodes: nodes,
            edges: edges,
            onNodesChange: onNodesChange,
            onEdgesChange: onEdgesChange,
            nodeTypes: nodeTypes,
            fitView: true,
            minZoom: 0.1,
            maxZoom: 2,
            panOnDrag: true,
            panOnScroll: false,
            zoomOnScroll: true,
            zoomOnPinch: true,
            zoomOnDoubleClick: false,
            nodesDraggable: true,
            nodesConnectable: false,
            elementsSelectable: true,
            selectNodesOnDrag: false,
            proOptions: { hideAttribution: true },
            defaultEdgeOptions: { type: 'smoothstep', animated: true, style: { stroke: '#ff8c00', strokeWidth: 2 } }
          },
            React.createElement(Controls, { showInteractive: false, showZoom: true, showFitView: true })
          )
        );
      }
      
      function MindMapApp({ initialData, title }) {
        return React.createElement(ReactFlowProvider, null,
          React.createElement(MindMapContent, { initialData: initialData, title: title })
        );
      }
      
      const treeData = ${treeDataJson};
      const title = "${escapedTitle}";
      const root = ReactDOM.createRoot(document.getElementById('mindmap-root'));
      root.render(React.createElement(MindMapApp, { initialData: treeData, title: title }));
    })();
  </script>
</body>
</html>`
}



// add node functionality //
/*
React.createElement('div', { className: 'node-actions' },
            React.createElement('button', { className: 'action-btn', onClick: handleAddChild, title: 'Add child' }, '+'),
            !data.isRoot && React.createElement('button', { className: 'action-btn', onClick: handleDelete, title: 'Delete' }, '×')
          ),
*/