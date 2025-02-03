import{j as e,a as n,T as l,G as a,o as r,p as k,I as w,V as K,q as W,B as D}from"./@mui-926ab118.js";import{r as s}from"./react-4cc0034d.js";import{a as R}from"./axios-1779699b.js";import"./clsx-0839fdbe.js";import"./react-transition-group-dd4db10c.js";import"./@babel-591e0485.js";import"./react-dom-502af1e5.js";import"./scheduler-765c72db.js";import"./@emotion-4d3fcc43.js";import"./hoist-non-react-statics-23d96a9a.js";import"./react-is-e8e5dbb3.js";import"./stylis-79144faa.js";const j="/assets/adminpage-25b220be.jpg",H=({onLogin:g})=>{const[o,b]=s.useState(""),[c,v]=s.useState(""),[m,y]=s.useState(""),[d,x]=s.useState(""),[u,f]=s.useState(""),[h,N]=s.useState(""),[i,S]=s.useState(!1),[p,A]=s.useState(""),C=async t=>{t.preventDefault();const I={bucketName:o,cloudFrontDomain:c,cloudAccessKeyId:m,secretAccessKey:d,regionName:u,rtspUrl:h};try{await R.post("http://localhost:5000/save-admin-details",I),g("/admin-dashboard")}catch{A("Error saving admin details")}};return e("div",{className:"flex min-h-screen",children:e("div",{className:"flex-1 flex items-center justify-center",style:{backgroundImage:`url(${j})`,backgroundSize:"cover",backgroundPosition:"center"},children:n("div",{className:"backdrop-blur-md bg-white bg-opacity-50 p-8 rounded-lg shadow-lg max-w-md w-full",children:[e(l,{variant:"h4",component:"h2",className:"mb-6 text-center text-gray-800",children:"Admin Login"}),p&&e(l,{color:"error",className:"mb-4 text-center",children:p}),n("form",{onSubmit:C,className:"space-y-6",children:[e(l,{variant:"h6",className:"text-gray-700 text-center pt-10",children:"Cloud Settings"}),n(a,{container:!0,spacing:2,children:[e(a,{item:!0,xs:12,sm:6,children:e(r,{label:"Bucket Name",variant:"outlined",fullWidth:!0,value:o,onChange:t=>b(t.target.value),className:"mb-4"})}),e(a,{item:!0,xs:12,sm:6,children:e(r,{label:"CloudFront Domain URL",variant:"outlined",fullWidth:!0,value:c,onChange:t=>v(t.target.value),className:"mb-4"})})]}),n(a,{container:!0,spacing:2,children:[e(a,{item:!0,xs:12,sm:6,children:e(r,{label:"Cloud Access Key ID",variant:"outlined",fullWidth:!0,value:m,onChange:t=>y(t.target.value),className:"mb-4"})}),e(a,{item:!0,xs:12,sm:6,children:e(r,{label:"Secret Access Key",variant:"outlined",fullWidth:!0,type:i?"text":"password",value:d,onChange:t=>x(t.target.value),InputProps:{endAdornment:e(k,{position:"end",children:e(w,{onClick:()=>S(!i),edge:"end",children:i?e(K,{}):e(W,{})})})},className:"mb-4"})})]}),n(a,{container:!0,spacing:2,children:[e(a,{item:!0,xs:12,sm:6,children:e(r,{label:"Region Name",variant:"outlined",fullWidth:!0,value:u,onChange:t=>f(t.target.value),className:"mb-4"})}),e(a,{item:!0,xs:12,sm:6,children:e(r,{label:"RTSP URL",variant:"outlined",fullWidth:!0,value:h,onChange:t=>N(t.target.value),className:"mb-4"})})]}),e(D,{type:"submit",variant:"contained",color:"primary",fullWidth:!0,className:"py-2",children:"Sign in"})]})]})})})};export{H as default};
