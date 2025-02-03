import{r,j as t}from"./react-BmcpohRg.js";import{J as P}from"./jszip-BdL31gVH.js";import{B as d,b as v,C as R}from"./@mui-lFRa_zgC.js";import"./clsx-B-dksMZM.js";import"./react-transition-group-Dykj5AjV.js";import"./@babel-DuB8yAtz.js";import"./react-dom-d0Lv2CMa.js";import"./scheduler-CzFDRTuY.js";import"./@emotion-Bi68Cbrn.js";import"./hoist-non-react-statics-Buxaj0Kl.js";import"./react-is-8JwjhRSi.js";import"./stylis-YPZU7XtI.js";const F=()=>{const[i,g]=r.useState([]),[h,y]=r.useState(""),[m,k]=r.useState(""),[u,j]=r.useState([]),[s,p]=r.useState(null),f=r.useRef(null),b=r.useRef(null),w="https://cors-anywhere.herokuapp.com/",C=async()=>{const e=h.split("T")[0],o=h.split("T")[1],n=m.split("T")[0],c=m.split("T")[1],l=`http://127.0.0.1:5000/videos?start_date=${e}&start_time=${o}&end_date=${n}&end_time=${c}`;try{const x=await(await fetch(l)).json();g(x.s3_videos||[]),x.s3_videos.length>0?p(x.s3_videos[0]):g([])}catch(a){console.error("Error fetching videos:",a)}},S=e=>{j(o=>o.includes(e)?o.filter(n=>n!==e):[...o,e])},T=async()=>{const e=new P;for(const o of u)try{const c=await(await fetch(w+o)).blob(),l=o.split("/").pop();e.file(l,c)}catch(n){console.error("Error downloading video:",n)}e.generateAsync({type:"blob"}).then(o=>{const n=document.createElement("a");n.href=URL.createObjectURL(o),n.download="videos.zip",document.body.appendChild(n),n.click(),document.body.removeChild(n)})},V=()=>{const e=i.indexOf(s);e!==-1&&e<i.length-1&&p(i[e+1])},D=e=>{p(e)},N=()=>{if(s){const e=document.createElement("a");e.href=s,e.download=s.split("/").pop(),e.click()}},E=()=>{const e=f.current,o=b.current;if(e&&o){const n=o.getContext("2d");if(e.readyState===4){o.width=e.videoWidth,o.height=e.videoHeight,n.drawImage(e,0,0,o.width,o.height);const c=o.toDataURL("image/png"),l=new Date().toLocaleString().replace(/[/: ]/g,"_"),a=document.createElement("a");a.href=c,a.download=`snapshot_${l}.png`,a.click()}else console.error("Video is not ready for snapshot")}};return t.jsxs("div",{className:"flex min-h-screen bg-white",children:[t.jsxs("div",{className:"w-[70%] flex flex-col items-center justify-center text-gray-500 text-lg bg-gray-200",children:[t.jsx("h1",{className:"mb-4 text-xl font-semibold",children:" Playback Video Player"}),t.jsx("div",{children:s?t.jsx("video",{ref:f,width:"1020",height:"720",controls:!0,autoPlay:!0,muted:!0,crossOrigin:"anonymous",src:s,onEnded:V}):t.jsx("p",{className:"text-gray-500",children:"Click on a video to play it."})}),s&&t.jsx(d,{variant:"contained",color:"secondary",sx:{marginTop:"10px",backgroundColor:"#1976d2",color:"white",height:"48px","&:hover":{backgroundColor:"#1565c0"}},onClick:N,children:"Download Video"})]}),t.jsxs("div",{className:"w-[30%] p-8 space-y-6 bg-gray-100",children:[t.jsxs("div",{children:[t.jsx("label",{className:"block text-gray-700 font-medium mb-2",children:"Start Date Time:"}),t.jsx(v,{type:"datetime-local",value:h,onChange:e=>y(e.target.value),variant:"outlined",fullWidth:!0,InputLabelProps:{shrink:!0},sx:{backgroundColor:"#f0f0f0"}})]}),t.jsxs("div",{children:[t.jsx("label",{className:"block text-gray-700 font-medium mb-2",children:"End Date Time:"}),t.jsx(v,{type:"datetime-local",value:m,onChange:e=>k(e.target.value),variant:"outlined",fullWidth:!0,InputLabelProps:{shrink:!0},sx:{backgroundColor:"#f0f0f0"}})]}),t.jsx(d,{variant:"contained",color:"primary",fullWidth:!0,sx:{backgroundColor:"#1976d2",color:"white",height:"48px","&:hover":{backgroundColor:"#1565c0"}},onClick:C,children:"SHOW VIDEO"}),t.jsx(d,{variant:"contained",color:"primary",fullWidth:!0,sx:{backgroundColor:"#1976d2",color:"white",height:"48px","&:hover":{backgroundColor:"#1565c0"}},onClick:E,children:"TAKE SNAPSHOT"}),t.jsxs("div",{children:[t.jsx("h2",{className:"text-lg font-semibold text-gray-700",children:"Recorded Videos"}),t.jsx("ul",{id:"s3-video-list",className:"mt-4 space-y-2",children:i.length>0?i.map((e,o)=>t.jsxs("li",{children:[t.jsx(R,{checked:u.includes(e),onChange:()=>S(e)}),t.jsx("button",{className:"text-blue-500 underline",onClick:()=>D(e),children:e.split("/").pop()})]},o)):t.jsx("li",{className:"text-gray-500",children:"No videos found for the selected date range."})}),u.length>0&&t.jsx(d,{variant:"contained",fullWidth:!0,onClick:T,sx:{backgroundColor:"#1976d2",color:"white",height:"48px","&:hover":{backgroundColor:"#1565c0"}},children:"Download Selected Videos as ZIP"})]})]}),t.jsx("canvas",{ref:b,style:{display:"none"}})]})};export{F as default};
