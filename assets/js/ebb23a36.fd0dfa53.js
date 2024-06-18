"use strict";(self.webpackChunkdocusaurus_tsx=self.webpackChunkdocusaurus_tsx||[]).push([[761],{8566:(e,n,d)=>{d.r(n),d.d(n,{assets:()=>l,contentTitle:()=>o,default:()=>h,frontMatter:()=>t,metadata:()=>r,toc:()=>a});var i=d(4848),s=d(8453);const t={},o="How do I use Pretrained embeddings (e.g. GloVe)?",r={id:"FAQ/pretrained_embeddings",title:"How do I use Pretrained embeddings (e.g. GloVe)?",description:"This is handled in the initial steps of the onmt_train execution.",source:"@site/docs/FAQ/pretrained_embeddings.md",sourceDirName:"FAQ",slug:"/FAQ/pretrained_embeddings",permalink:"/eole/docs/FAQ/pretrained_embeddings",draft:!1,unlisted:!1,editUrl:"https://github.com/eole-nlp/eole/tree/main/docs/docs/FAQ/pretrained_embeddings.md",tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Position encoding: Absolute vs Relative vs Rotary Embeddings vs Alibi",permalink:"/eole/docs/FAQ/position_encoding"},next:{title:"What special tokens are used?",permalink:"/eole/docs/FAQ/special_tokens"}},l={},a=[{value:"Example",id:"example",level:3}];function c(e){const n={code:"code",h1:"h1",h3:"h3",li:"li",ol:"ol",p:"p",pre:"pre",ul:"ul",...(0,s.R)(),...e.components};return(0,i.jsxs)(i.Fragment,{children:[(0,i.jsx)(n.h1,{id:"how-do-i-use-pretrained-embeddings-eg-glove",children:"How do I use Pretrained embeddings (e.g. GloVe)?"}),"\n",(0,i.jsxs)(n.p,{children:["This is handled in the initial steps of the ",(0,i.jsx)(n.code,{children:"onmt_train"})," execution."]}),"\n",(0,i.jsx)(n.p,{children:"Pretrained embeddings can be configured in the main YAML configuration file."}),"\n",(0,i.jsx)(n.h3,{id:"example",children:"Example"}),"\n",(0,i.jsxs)(n.ol,{children:["\n",(0,i.jsx)(n.li,{children:"Get GloVe files:"}),"\n"]}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-bash",children:'mkdir "glove_dir"\nwget http://nlp.stanford.edu/data/glove.6B.zip\nunzip glove.6B.zip -d "glove_dir"\n'})}),"\n",(0,i.jsxs)(n.ol,{start:"2",children:["\n",(0,i.jsx)(n.li,{children:"Adapt the configuration:"}),"\n"]}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-yaml",children:'# <your_config>.yaml\n\n<Your data config...>\n\n...\n\n# this means embeddings will be used for both encoder and decoder sides\nboth_embeddings: glove_dir/glove.6B.100d.txt\n# to set src and tgt embeddings separately:\n# src_embeddings: ...\n# tgt_embeddings: ...\n\n# supported types: GloVe, word2vec\nembeddings_type: "GloVe"\n\n# word_vec_size need to match with the pretrained embeddings dimensions\nword_vec_size: 100\n\n'})}),"\n",(0,i.jsxs)(n.ol,{start:"3",children:["\n",(0,i.jsx)(n.li,{children:"Train:"}),"\n"]}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-bash",children:"onmt_train -config <your_config>.yaml\n"})}),"\n",(0,i.jsx)(n.p,{children:"Notes:"}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:["the matched embeddings will be saved at ",(0,i.jsx)(n.code,{children:"<save_data>.enc_embeddings.pt"})," and ",(0,i.jsx)(n.code,{children:"<save_data>.dec_embeddings.pt"}),";"]}),"\n",(0,i.jsxs)(n.li,{children:["additional flags ",(0,i.jsx)(n.code,{children:"freeze_word_vecs_enc"})," and ",(0,i.jsx)(n.code,{children:"freeze_word_vecs_dec"})," are available to freeze the embeddings."]}),"\n"]})]})}function h(e={}){const{wrapper:n}={...(0,s.R)(),...e.components};return n?(0,i.jsx)(n,{...e,children:(0,i.jsx)(c,{...e})}):c(e)}},8453:(e,n,d)=>{d.d(n,{R:()=>o,x:()=>r});var i=d(6540);const s={},t=i.createContext(s);function o(e){const n=i.useContext(t);return i.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function r(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(s):e.components||s:o(e.components),i.createElement(t.Provider,{value:n},e.children)}}}]);