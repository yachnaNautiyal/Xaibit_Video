import{r,j as t}from"./react-BmcpohRg.js";import{d as x,M as l,B as n}from"./@mui-lFRa_zgC.js";import"./clsx-B-dksMZM.js";import"./react-transition-group-Dykj5AjV.js";import"./@babel-DuB8yAtz.js";import"./react-dom-d0Lv2CMa.js";import"./scheduler-CzFDRTuY.js";import"./@emotion-Bi68Cbrn.js";import"./hoist-non-react-statics-Buxaj0Kl.js";import"./react-is-8JwjhRSi.js";import"./stylis-YPZU7XtI.js";const N=()=>{const[a,d]=r.useState(null),[p,i]=r.useState([]);r.useState(""),r.useState("");const[h,m]=r.useState(""),c=async()=>{try{const e=await fetch("/video_feed",{cache:"no-store"});if(e.ok){const s=await e.blob(),o=URL.createObjectURL(s);a&&URL.revokeObjectURL(a),d(o)}else console.error("No video available:",e.statusText)}catch(e){console.error("Error fetching video:",e),setTimeout(c,2e4)}},u=async()=>{try{const e=await fetch("http://localhost:5000/video_feed"),s=e.headers.get("Content-Type");if(s&&s.includes("application/json")){const o=await e.json();o.s3_videos?i(o.s3_videos):i([])}else{const o=await e.text();console.error("Expected JSON but received:",o)}}catch(e){console.error("Error fetching videos:",e)}};return r.useEffect(()=>{c()},[]),t.jsxs("div",{className:"flex min-h-screen bg-white",children:[t.jsxs("div",{className:"w-[70%] flex flex-col items-center justify-center text-gray-500 text-lg bg-gray-200",children:[t.jsx("h1",{className:"mb-4 text-xl font-semibold",children:"Live Video Feed"}),t.jsx("img",{src:"http://localhost:5000/video_feed",alt:"Live Video Feed",width:"640",height:"480"})]}),t.jsxs("div",{className:"w-[30%] p-8 space-y-6 bg-gray-100",children:[t.jsxs("div",{children:[t.jsx("label",{className:"block text-gray-700 font-medium mb-2",children:"Select the Camera:"}),t.jsxs(x,{value:h,onChange:e=>m(e.target.value),variant:"outlined",fullWidth:!0,sx:{backgroundColor:"#f0f0f0"},children:[t.jsx(l,{value:"camera1",children:"Camera 1"}),t.jsx(l,{value:"camera2",children:"Camera 2"})]})]}),t.jsx(n,{variant:"contained",color:"primary",fullWidth:!0,sx:{backgroundColor:"#1976d2",color:"white",height:"48px","&:hover":{backgroundColor:"#1565c0"}},onClick:u,children:"Show Live Video"}),t.jsx(n,{variant:"contained",color:"primary",fullWidth:!0,sx:{backgroundColor:"#1976d2",color:"white",height:"48px","&:hover":{backgroundColor:"#1565c0"}},children:"Take Snapshot"}),t.jsx("div",{})]})]})};export{N as default};
