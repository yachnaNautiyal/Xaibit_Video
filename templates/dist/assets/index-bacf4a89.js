import{j as e,a as s,B as P,I as A,M as I,D as R,L as D,b as v,d as y,e as b,T as w,f as C,h as O,A as k,P as B,i as V,k as T,l as S,m as F,F as z}from"./@mui-926ab118.js";import{r as l}from"./react-4cc0034d.js";import{c as M}from"./react-dom-502af1e5.js";import{u as N,A as j}from"./@auth0-6f045022.js";import{L as c,B as W}from"./react-router-dom-6e1dbecc.js";import{d as $,e as q,f as m,b as K}from"./react-router-b1388c1d.js";import"./clsx-0839fdbe.js";import"./react-transition-group-dd4db10c.js";import"./@babel-591e0485.js";import"./@emotion-4d3fcc43.js";import"./hoist-non-react-statics-23d96a9a.js";import"./react-is-e8e5dbb3.js";import"./stylis-79144faa.js";import"./scheduler-765c72db.js";import"./@remix-run-f98ef440.js";(function(){const n=document.createElement("link").relList;if(n&&n.supports&&n.supports("modulepreload"))return;for(const r of document.querySelectorAll('link[rel="modulepreload"]'))d(r);new MutationObserver(r=>{for(const t of r)if(t.type==="childList")for(const o of t.addedNodes)o.tagName==="LINK"&&o.rel==="modulepreload"&&d(o)}).observe(document,{childList:!0,subtree:!0});function a(r){const t={};return r.integrity&&(t.integrity=r.integrity),r.referrerPolicy&&(t.referrerPolicy=r.referrerPolicy),r.crossOrigin==="use-credentials"?t.credentials="include":r.crossOrigin==="anonymous"?t.credentials="omit":t.credentials="same-origin",t}function d(r){if(r.ep)return;r.ep=!0;const t=a(r);fetch(r.href,t)}})();const U="modulepreload",H=function(i){return"/"+i},L={},u=function(n,a,d){if(!a||a.length===0)return n();const r=document.getElementsByTagName("link");return Promise.all(a.map(t=>{if(t=H(t),t in L)return;L[t]=!0;const o=t.endsWith(".css"),E=o?'[rel="stylesheet"]':"";if(!!d)for(let p=r.length-1;p>=0;p--){const f=r[p];if(f.href===t&&(!o||f.rel==="stylesheet"))return}else if(document.querySelector(`link[href="${t}"]${E}`))return;const h=document.createElement("link");if(h.rel=o?"stylesheet":U,o||(h.as="script",h.crossOrigin=""),h.href=t,document.head.appendChild(h),o)return new Promise((p,f)=>{h.addEventListener("load",p),h.addEventListener("error",()=>f(new Error(`Unable to preload CSS for ${t}`)))})})).then(()=>n()).catch(t=>{const o=new Event("vite:preloadError",{cancelable:!0});if(o.payload=t,window.dispatchEvent(o),!o.defaultPrevented)throw t})};function X(){return e("footer",{className:"bg-gray-800 text-white py-4",children:e("div",{className:"max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center",children:e("p",{className:"text-sm",children:"© 2024 My Website. All rights reserved."})})})}class Y extends l.Component{constructor(n){super(n),this.state={hasError:!1}}static getDerivedStateFromError(n){return{hasError:!0}}componentDidCatch(n,a){console.error("ErrorBoundary caught an error",n,a)}render(){return this.state.hasError?e("h1",{children:"Something went wrong."}):this.props.children}}const g=({children:i})=>{const{isAuthenticated:n,isLoading:a}=N();return a?e("div",{children:"Loading..."}):n?i:e($,{to:"/login-options"})},x="/assets/logo-8931d7a0.png";function _(){return e("nav",{className:"bg-white shadow-lg",children:e("div",{className:"max-w-7xl mx-auto px-2 sm:px-6 lg:px-8",children:s("div",{className:"relative flex items-center justify-between h-16",children:[e("div",{className:"flex items-center flex-shrink-0",children:s("a",{href:"/",children:[e("img",{className:"block lg:hidden h-8 w-auto",src:x,alt:"Logo"}),e("img",{className:"hidden lg:block h-12 w-auto",src:x,alt:"Logo"})]})}),e("div",{className:"hidden sm:block sm:ml-6",children:s("div",{className:"flex space-x-4",children:[e(c,{to:"/services",className:"text-blue-900 px-3 py-2 rounded-md text-sm font-medium hover:bg-blue-100 transition-colors duration-300",children:"Services"}),e(c,{to:"/industry",className:"text-blue-900 px-3 py-2 rounded-md text-sm font-medium hover:bg-blue-100 transition-colors duration-300",children:"Industry"}),e(c,{to:"/testimonials",className:"text-blue-900 px-3 py-2 rounded-md text-sm font-medium hover:bg-blue-100 transition-colors duration-300",children:"Testimonials"}),e(c,{to:"/about",className:"text-blue-900 px-3 py-2 rounded-md text-sm font-medium hover:bg-blue-100 transition-colors duration-300",children:"About"}),e(c,{to:"/blog",className:"text-blue-900 px-3 py-2 rounded-md text-sm font-medium hover:bg-blue-100 transition-colors duration-300",children:"Blog"}),e(c,{to:"/contact",className:"text-blue-900 px-3 py-2 rounded-md text-sm font-medium hover:bg-blue-100 transition-colors duration-300",children:"Contact"}),e(c,{to:"/partners",className:"text-blue-900 px-3 py-2 rounded-md text-sm font-medium hover:bg-blue-100 transition-colors duration-300",children:"Partners"}),e(c,{to:"/book-meeting",className:"text-blue-900 px-3 py-2 rounded-md text-sm font-medium hover:bg-blue-100 transition-colors duration-300",children:"Book a Meeting"}),e(P,{variant:"contained",color:"primary",onClick:()=>{window.open("/login-options","_blank")},className:"bg-blue-900 text-white px-3 py-2 rounded-md text-sm font-medium transform hover:scale-105 transition-transform duration-300",children:"Login"})]})})]})})})}const G=()=>{const{isAuthenticated:i,logout:n}=N(),[a,d]=l.useState(!1),r=o=>()=>{d(o)},t=[{text:"Admin Login",path:"/AdminLoginForm",icon:e(k,{})},{text:"Live Video",path:"/CameraView",icon:e(B,{})},{text:"PlayBack Video",path:"/VideoPlayer",icon:e(V,{})},{text:"New Registration",path:"/new-registration",icon:e(T,{})},{text:"Personal Details",path:"/main-program",icon:e(S,{})},{text:"Data Summary",path:"/data-analysis",icon:e(F,{})}];return e("nav",{className:"bg-white shadow-lg",children:s("div",{className:"max-w-7xl mx-auto px-2 sm:px-6 lg:px-8 flex justify-between items-center h-16",children:[e("div",{className:"flex items-center",children:e(c,{to:"/",children:e("img",{className:"h-10 w-auto",src:x,alt:"Logo"})})}),i&&s("div",{className:"flex items-center",children:[e(A,{edge:"start",color:"inherit","aria-label":"menu",onClick:r(!0),children:e(I,{})}),e(R,{anchor:"right",open:a,onClose:r(!1),children:e("div",{className:"w-64",role:"presentation",onClick:r(!1),onKeyDown:r(!1),children:s(D,{children:[t.map(o=>s(v,{button:!0,component:c,to:o.path,children:[e(y,{children:o.icon}),e(b,{primary:e(w,{variant:"h6",sx:{fontWeight:500},children:o.text})})]},o.text)),e(C,{sx:{my:2}}),s(v,{button:!0,onClick:()=>n({returnTo:window.location.origin}),children:[e(y,{children:e(O,{})}),e(b,{primary:e(w,{variant:"h6",sx:{fontWeight:500},children:"Logout"})})]})]})})})]})]})})},J=l.lazy(()=>u(()=>import("./homepage-77e66932.js"),["assets/homepage-77e66932.js","assets/@mui-926ab118.js","assets/react-4cc0034d.js","assets/clsx-0839fdbe.js","assets/react-transition-group-dd4db10c.js","assets/@babel-591e0485.js","assets/react-dom-502af1e5.js","assets/scheduler-765c72db.js","assets/@emotion-4d3fcc43.js","assets/hoist-non-react-statics-23d96a9a.js","assets/react-is-e8e5dbb3.js","assets/stylis-79144faa.js"])),Q=l.lazy(()=>u(()=>import("./LoginOptions-3b45b348.js"),["assets/LoginOptions-3b45b348.js","assets/@mui-926ab118.js","assets/react-4cc0034d.js","assets/clsx-0839fdbe.js","assets/react-transition-group-dd4db10c.js","assets/@babel-591e0485.js","assets/react-dom-502af1e5.js","assets/scheduler-765c72db.js","assets/@emotion-4d3fcc43.js","assets/hoist-non-react-statics-23d96a9a.js","assets/react-is-e8e5dbb3.js","assets/stylis-79144faa.js","assets/@auth0-6f045022.js","assets/react-router-b1388c1d.js","assets/@remix-run-f98ef440.js"])),Z=l.lazy(()=>u(()=>import("./newRegistration-4ecb9bbc.js"),["assets/newRegistration-4ecb9bbc.js","assets/@mui-926ab118.js","assets/react-4cc0034d.js","assets/clsx-0839fdbe.js","assets/react-transition-group-dd4db10c.js","assets/@babel-591e0485.js","assets/react-dom-502af1e5.js","assets/scheduler-765c72db.js","assets/@emotion-4d3fcc43.js","assets/hoist-non-react-statics-23d96a9a.js","assets/react-is-e8e5dbb3.js","assets/stylis-79144faa.js","assets/axios-1779699b.js"])),ee=l.lazy(()=>u(()=>import("./mainProgram-6b97643f.js"),["assets/mainProgram-6b97643f.js","assets/@mui-926ab118.js","assets/react-4cc0034d.js","assets/clsx-0839fdbe.js","assets/react-transition-group-dd4db10c.js","assets/@babel-591e0485.js","assets/react-dom-502af1e5.js","assets/scheduler-765c72db.js","assets/@emotion-4d3fcc43.js","assets/hoist-non-react-statics-23d96a9a.js","assets/react-is-e8e5dbb3.js","assets/stylis-79144faa.js"])),te=l.lazy(()=>u(()=>import("./BigdataAnalysis-cf124acd.js"),["assets/BigdataAnalysis-cf124acd.js","assets/@mui-926ab118.js","assets/react-4cc0034d.js","assets/clsx-0839fdbe.js","assets/react-transition-group-dd4db10c.js","assets/@babel-591e0485.js","assets/react-dom-502af1e5.js","assets/scheduler-765c72db.js","assets/@emotion-4d3fcc43.js","assets/hoist-non-react-statics-23d96a9a.js","assets/react-is-e8e5dbb3.js","assets/stylis-79144faa.js","assets/axios-1779699b.js","assets/react-datepicker-8937daa9.js","assets/@floating-ui-07943c42.js","assets/react-datepicker-e4868022.css","assets/react-chartjs-2-fa95bfbc.js","assets/chart.js-3686957a.js","assets/@kurkle-36ca2f10.js","assets/date-fns-66ee9ebe.js"])),re=l.lazy(()=>u(()=>import("./AdminLoginForm-23ed2d3b.js"),["assets/AdminLoginForm-23ed2d3b.js","assets/@mui-926ab118.js","assets/react-4cc0034d.js","assets/clsx-0839fdbe.js","assets/react-transition-group-dd4db10c.js","assets/@babel-591e0485.js","assets/react-dom-502af1e5.js","assets/scheduler-765c72db.js","assets/@emotion-4d3fcc43.js","assets/hoist-non-react-statics-23d96a9a.js","assets/react-is-e8e5dbb3.js","assets/stylis-79144faa.js","assets/axios-1779699b.js"])),oe=l.lazy(()=>u(()=>import("./CameraView-f3bd1250.js"),["assets/CameraView-f3bd1250.js","assets/@mui-926ab118.js","assets/react-4cc0034d.js","assets/clsx-0839fdbe.js","assets/react-transition-group-dd4db10c.js","assets/@babel-591e0485.js","assets/react-dom-502af1e5.js","assets/scheduler-765c72db.js","assets/@emotion-4d3fcc43.js","assets/hoist-non-react-statics-23d96a9a.js","assets/react-is-e8e5dbb3.js","assets/stylis-79144faa.js"])),ne=l.lazy(()=>u(()=>import("./VideoPlayer-26fb3824.js"),["assets/VideoPlayer-26fb3824.js","assets/@mui-926ab118.js","assets/react-4cc0034d.js","assets/clsx-0839fdbe.js","assets/react-transition-group-dd4db10c.js","assets/@babel-591e0485.js","assets/react-dom-502af1e5.js","assets/scheduler-765c72db.js","assets/@emotion-4d3fcc43.js","assets/hoist-non-react-statics-23d96a9a.js","assets/react-is-e8e5dbb3.js","assets/stylis-79144faa.js","assets/jszip-edafff3e.js"])),ie=({children:i})=>{const n=K(),d=["/AdminLoginForm","/new-registration","/main-program","/data-analysis","/CameraView","/VideoPlayer"].includes(n.pathname);return s(z,{children:[n.pathname==="/"?e(_,{}):d?e(G,{}):e(_,{}),i]})},ae=()=>e(W,{children:s(Y,{children:[s(ie,{children:[" ",e(l.Suspense,{fallback:e("div",{className:"flex justify-center items-center h-screen",children:"Loading..."}),children:s(q,{children:[e(m,{path:"/",element:e(J,{})}),e(m,{path:"/AdminLoginForm",element:e(re,{})}),e(m,{path:"/login-options",element:e(Q,{})}),e(m,{path:"/new-registration",element:e(g,{children:e(Z,{})})}),e(m,{path:"/main-program",element:e(g,{children:e(ee,{})})}),e(m,{path:"/data-analysis",element:e(g,{children:e(te,{})})}),e(m,{path:"/CameraView",element:e(g,{children:e(oe,{})})}),e(m,{path:"/VideoPlayer",element:e(g,{children:e(ne,{})})})]})})]}),e(X,{})]})}),se=M(document.getElementById("root")),le=i=>{window.location.href="http://localhost:5000/login-options"};se.render(e(j,{domain:"dev-rhonptq1x6sjtesv.us.auth0.com",clientId:"mYxPguavO6k6M7w6t2ducK5C1ALfrXLb",authorizationParams:{redirect_uri:window.location.origin},onRedirectCallback:le,children:e(ae,{})}));
