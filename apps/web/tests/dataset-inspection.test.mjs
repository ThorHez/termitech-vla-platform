import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';
const fixtures=JSON.parse(readFileSync(new URL('./fixtures/dataset-inspections.json',import.meta.url)));
const server=await createServer({root:fileURLToPath(new URL('..',import.meta.url)),server:{middlewareMode:true,hmr:false,ws:false},optimizeDeps:{noDiscovery:true,include:[]},appType:'custom'});
try {
 const h=await server.ssrLoadModule('/src/datasetInspectionHelpers.ts');
 const {emptyDatasetProfile}=await server.ssrLoadModule('/src/DatasetProfileForm.tsx');
 const {draftComposedProcessing}=await server.ssrLoadModule('/src/composedProcessing.ts');
 const {DatasetInspection}=await server.ssrLoadModule('/src/DatasetInspection.tsx');
 const clone=structuredClone;
 const state=(initial=emptyDatasetProfile(),registered=false)=>({initial:clone(initial),registered,touched:new Set(),automatic:new Map()});
 const apply=(profile,inspection=fixtures.complete,s=state())=>h.mergeDatasetInspection(profile,inspection,s);
 await test('real Agent complete DTO fills actual keys and metadata while new placeholders can become Joint',()=>{
  const source=emptyDatasetProfile(),result=apply(source).profile;
  assert.equal(result.robot_type,'ARX5');assert.equal(result.action_representation,'joint');assert.equal(result.arm_mode,'single');
  assert.equal(result.action_dim,7);assert.equal(result.state_dim,7);assert.equal(result.fps,30);assert.equal(result.processing,undefined);
  assert.equal(result.cameras[0].key,'observation.images.left_wrist');assert.match(result.task_description,/红色方块/);
  assert.deepEqual(result.feature_keys,fixtures.complete.features.map(f=>f.key));assert.equal(source.robot_type,'');
  assert.equal(result.action_mode,'unknown');assert.equal(result.rotation_representation,'unknown');
 });
 await test('real ambiguous DTO never invents arms, EEF, robot or camera layout from dimensions',()=>{
  const result=apply(emptyDatasetProfile(),fixtures.ambiguous).profile;
  assert.equal(result.action_dim,20);assert.equal(result.robot_type,'');assert.equal(result.arm_mode,'single');
  assert.equal(result.action_representation,'unknown');assert.equal(result.action_mode,'unknown');
  assert.equal(fixtures.ambiguous.suggestions.action_representation,undefined);assert.equal(fixtures.ambiguous.processing.action_layout,undefined);
 });
 await test('history inspection labels source fields without creating or modifying processing',()=>{
  const profile=emptyDatasetProfile();profile.processing=draftComposedProcessing(profile);const before=clone(profile.processing);
  const result=apply(profile,fixtures.history).profile;
  assert.equal(result.arm_mode,'bimanual');assert.deepEqual(result.processing,before);assert.equal(result.state_dim,undefined);
  assert.equal(result.field_annotations.length,7);assert.ok(result.field_annotations.some(row=>row.role==='state'&&row.shape[0]===2));
  assert.ok(result.field_annotations.every(row=>!('normalize'in row)&&!('slot'in row)&&!('width'in row)));
  assert.equal(result.field_annotations.find(row=>row.role==='action').shape.at(-1),20);
 });
 await test('manual values and edits made and then cleared during a read stay protected',()=>{
  const initial=emptyDatasetProfile(),s=state(initial);let draft={...initial,robot_type:'custom',fps:91,task_description:'hand written'};
  h.recordInspectionEdits(s,initial,draft);const before=clone(draft);draft={...draft,task_description:''};h.recordInspectionEdits(s,before,draft);
  const result=apply(draft,fixtures.complete,s);assert.equal(result.profile.robot_type,'custom');assert.equal(result.profile.fps,91);assert.equal(result.profile.task_description,'');
  assert.ok(result.kept.includes('task_description'));
 });
 await test('an explicitly touched unchanged enum is no longer a new-profile placeholder',()=>{
  const profile=emptyDatasetProfile(),s=state(profile);s.touched.add('arm_mode');s.touched.add('action_representation');
  const result=apply(profile,fixtures.history,s).profile;assert.equal(result.arm_mode,'single');assert.equal(result.action_representation,'unknown');
 });
 await test('registered profile values remain authoritative and only genuinely missing dimensions are added',()=>{
  const profile={...emptyDatasetProfile(),robot_type:'registered robot',arm_mode:'bimanual',action_representation:'eef',cameras:[{name:'registered',key:'registered.image',kind:'depth'}],task_description:'registered task'};
  const result=apply(profile,fixtures.complete,state(profile,true)).profile;
  for(const key of ['robot_type','arm_mode','action_representation','cameras','task_description','action_mode','rotation_representation'])assert.deepEqual(result[key],profile[key]);
  assert.equal(result.fps,30);assert.equal(result.action_dim,7);
 });
 await test('existing registered component rows and each manual component edit are preserved',()=>{
  const profile=emptyDatasetProfile();profile.processing=draftComposedProcessing(profile);profile.processing.state_fields=[{key:'custom.state',width:11,arm:'shared',kind:'other',normalize:false}];
  profile.processing.action_layout=[{arm:'left',kind:'joint',width:9,normalize:false}];profile.processing.state_history=3;
  const s=state(profile);s.touched.add('processing.state_history');const result=apply(profile,fixtures.history,s).profile;
  assert.deepEqual(result.processing.state_fields,profile.processing.state_fields);assert.deepEqual(result.processing.action_layout,profile.processing.action_layout);assert.equal(result.processing.state_history,3);
  const registered=apply(profile,fixtures.complete,state(profile,true)).profile;assert.deepEqual(registered.processing,profile.processing);
 });
 await test('a legacy processing draft remains unchanged when source metadata is read again',()=>{
  const initial=emptyDatasetProfile(),s=state(initial);const first=apply(initial,fixtures.complete,s).profile;
  const enabled={...first,processing:draftComposedProcessing(first)};h.recordInspectionEdits(s,first,enabled);
  const result=apply(enabled,fixtures.complete,s).profile;assert.deepEqual(result.processing,enabled.processing);
  assert.equal(result.field_annotations.length,2);assert.equal(result.field_annotations[0].value_mode,'unknown');
 });
 await test('field tags merge by key and preserve only the labels manually edited',()=>{
  const initial=emptyDatasetProfile(),s=state(initial);const first=apply(initial,fixtures.history,s).profile;
  const manual=clone(first);manual.field_annotations[0].value_mode='delta';h.recordInspectionEdits(s,first,manual);
  const dto=clone(fixtures.history);dto.suggestions.field_annotations.forEach(row=>row.value_mode='absolute');
  const result=apply(manual,dto,s).profile;assert.equal(result.field_annotations[0].value_mode,'delta');
  assert.ok(result.field_annotations.slice(1).every(row=>row.value_mode==='absolute'));
  const cleared=h.clearAutomaticInspectionDefaults(result,s);
  assert.equal(cleared.field_annotations.length,1);assert.equal(cleared.field_annotations[0].key,manual.field_annotations[0].key);
  assert.equal(cleared.field_annotations[0].value_mode,'delta');assert.ok(cleared.field_annotations[0].shape.length);
 });
 await test('explicit annotation row removal is not undone by another metadata response',()=>{
  const initial=emptyDatasetProfile(),s=state(initial);const first=apply(initial,fixtures.history,s).profile;
  const manual={...first,field_annotations:first.field_annotations.slice(1)};h.recordInspectionEdits(s,first,manual);
  assert.deepEqual(apply(manual,fixtures.history,s).profile.field_annotations,manual.field_annotations);
 });
 await test('older Agent DTOs still produce unknown source tags without copying a processor',()=>{
  const dto=clone(fixtures.history);delete dto.suggestions.field_annotations;
  dto.suggestions.state_dim=20;const result=apply(emptyDatasetProfile(),dto).profile;
  assert.equal(result.field_annotations.length,7);assert.ok(result.field_annotations.every(row=>row.value_mode==='unknown'));
  assert.equal(result.processing,undefined);assert.equal(result.state_dim,undefined);
  const manual={...emptyDatasetProfile(),state_dim:37};const s=state(manual);s.touched.add('state_dim');
  const preserved=apply(manual,dto,s);assert.equal(preserved.profile.state_dim,37);assert.ok(preserved.kept.includes('state_dim'));
 });
 await test('stored action horizon comes from each original one-step or packed shape, including older Agents',()=>{
  for(const [name,horizon]of[['complete',1],['history',16]])for(const legacy of[false,true]){
   const dto=clone(fixtures[name]);if(legacy)delete dto.suggestions.field_annotations;
   const result=apply(emptyDatasetProfile(),dto).profile;
   assert.equal(result.field_annotations.find(row=>row.role==='action').horizon,horizon);
   assert.ok(result.field_annotations.filter(row=>row.role==='state').every(row=>!('horizon'in row)));
  }
 });
 await test('older Agent standalone gripper facts seed opening tags without inventing a value mode',()=>{
  const dto=clone(fixtures.history);delete dto.suggestions.field_annotations;
  const initial=emptyDatasetProfile(),s=state(initial),first=apply(initial,dto,s).profile;
  const index=first.field_annotations.findIndex(row=>row.component==='gripper');assert.ok(index>=0);
  assert.equal(first.field_annotations[index].representation,'gripper');
  assert.equal(first.field_annotations[index].value_mode,'unknown');
  const manual=clone(first);manual.field_annotations[index].representation='unknown';
  h.recordInspectionEdits(s,first,manual);
  assert.equal(apply(manual,dto,s).profile.field_annotations[index].representation,'unknown');
  assert.equal(first.action_representation,dto.suggestions.action_representation||'unknown');
 });
 await test('source horizon never invents an axis or trusts a declaration that conflicts with actual feature shape',()=>{
  const dto=clone(fixtures.history),label=dto.suggestions.field_annotations.find(row=>row.role==='action');
  label.horizon=99;
  assert.equal(apply(emptyDatasetProfile(),dto).profile.field_annotations.find(row=>row.role==='action').horizon,undefined);
  label.horizon=16;assert.equal(apply(emptyDatasetProfile(),dto).profile.field_annotations.find(row=>row.role==='action').horizon,16);
  for(const shape of[[],[2,16,20],[0,20],[16,0]]){
   const changed=clone(dto);changed.features.find(row=>row.key===label.key).shape=shape;
   const row=apply(emptyDatasetProfile(),changed).profile.field_annotations.find(row=>row.key===label.key);
   assert.equal(row.horizon,undefined);
  }
  const stateLabel=dto.suggestions.field_annotations.find(row=>row.role==='state');stateLabel.horizon=3;
  assert.ok(!('horizon'in apply(emptyDatasetProfile(),dto).profile.field_annotations.find(row=>row.key===stateLabel.key)));
 });
 await test('manual source horizons survive repeat reads and scope clearing while untouched automatic rows clear',()=>{
  const initial=emptyDatasetProfile(),s=state(initial),first=apply(initial,fixtures.history,s).profile;
  const manual=clone(first),index=manual.field_annotations.findIndex(row=>row.role==='action');
  manual.field_annotations[index].horizon=8;h.recordInspectionEdits(s,first,manual);
  const repeated=apply(manual,fixtures.history,s);
  assert.equal(repeated.profile.field_annotations[index].horizon,8);
  assert.ok(repeated.kept.includes(`field_annotations[${index}].horizon`));
  const cleared=h.clearAutomaticInspectionDefaults(repeated.profile,s);
  assert.equal(cleared.field_annotations.length,1);assert.equal(cleared.field_annotations[0].horizon,8);
  const untouchedState=state(initial),automatic=apply(initial,fixtures.history,untouchedState).profile;
  assert.equal((h.clearAutomaticInspectionDefaults(automatic,untouchedState).field_annotations||[]).length,0);
 });
 await test('changing an action label to state cannot regain a horizon from a later metadata response',()=>{
  const initial=emptyDatasetProfile(),s=state(initial),first=apply(initial,fixtures.history,s).profile;
  const manual=clone(first),index=manual.field_annotations.findIndex(row=>row.role==='action');
  manual.field_annotations[index].role='state';delete manual.field_annotations[index].horizon;
  h.recordInspectionEdits(s,first,manual);
  const repeated=apply(manual,fixtures.history,s).profile;
  assert.equal(repeated.field_annotations[index].role,'state');assert.ok(!('horizon'in repeated.field_annotations[index]));
  const untouchedState=state(initial),automatic=apply(initial,fixtures.history,untouchedState).profile;
  const changed=clone(fixtures.history);changed.suggestions.field_annotations.find(row=>row.role==='action').role='state';
  const refreshed=apply(automatic,changed,untouchedState).profile;
  assert.equal(refreshed.field_annotations[index].role,'state');assert.ok(!('horizon'in refreshed.field_annotations[index]));
 });
 await test('a conflicting replacement response removes a stale automatic horizon rather than keeping an obsolete length',()=>{
  const initial=emptyDatasetProfile(),s=state(initial),first=apply(initial,fixtures.history,s).profile;
  const dto=clone(fixtures.history),action=dto.suggestions.field_annotations.find(row=>row.role==='action');
  dto.features.find(row=>row.key===action.key).shape=[8,20];action.shape=[8,20];action.horizon=16;
  const result=apply(first,dto,s).profile.field_annotations.find(row=>row.role==='action');
  assert.deepEqual(result.shape,[8,20]);assert.equal(result.horizon,undefined);
 });
 await test('legacy state dimensions require the last axis of an actual unified state feature',()=>{
  for(const key of ['state','observation.state']){
   const dto=clone(fixtures.history);delete dto.suggestions.field_annotations;
   dto.features.push({key,dtype:'float32',shape:[2,7],names:[]});dto.suggestions.state_dim=20;
   assert.equal(apply(emptyDatasetProfile(),dto).profile.state_dim,undefined);
   dto.suggestions.state_dim=7;assert.equal(apply(emptyDatasetProfile(),dto).profile.state_dim,7);
   dto.features.at(-1).shape=[];assert.equal(apply(emptyDatasetProfile(),dto).profile.state_dim,undefined);
   dto.features.at(-1).shape=[0];dto.suggestions.state_dim=0;
   assert.equal(apply(emptyDatasetProfile(),dto).profile.state_dim,undefined);
  }
 });
 await test('metadata action declarations may fill a new draft but inferred declarations cannot',()=>{
  const dto=clone(fixtures.complete);dto.suggestions.action_mode='delta';dto.suggestions.rotation_representation='none';
  dto.evidence.push({field:'action_mode',kind:'metadata',reason:'explicit',keys:[]},{field:'rotation_representation',kind:'metadata',reason:'explicit',keys:[]});
  const result=apply(emptyDatasetProfile(),dto).profile;assert.equal(result.action_mode,'delta');assert.equal(result.rotation_representation,'none');
 });
 await test('scope changes remove old untouched automatic values and preserve manual drafts',()=>{
  const initial=emptyDatasetProfile(),s=state(initial);const first=apply(initial,fixtures.complete,s).profile;
  const manual={...first,robot_type:'my robot'};h.recordInspectionEdits(s,first,manual);
  const cleared=h.clearAutomaticInspectionDefaults(manual,s);assert.equal(cleared.robot_type,'my robot');assert.equal(cleared.action_dim,undefined);assert.equal(cleared.task_description,'');assert.deepEqual(cleared.cameras,initial.cameras);
  const result=apply(cleared,fixtures.history,s).profile;assert.equal(result.robot_type,'my robot');assert.equal(result.action_dim,20);assert.equal(result.arm_mode,'bimanual');
 });
 await test('canonical paths and query identities reject late results from another node, directory or query',()=>{
  assert.equal(h.canonicalDatasetInspectionPath('/dataset//./one/'),'/dataset/one');assert.equal(h.canonicalDatasetInspectionPath('/'),'/');
  for(const path of ['relative','/a/../b','/a\n/b','/a\u0085/b'])assert.equal(h.canonicalDatasetInspectionPath(path),'');
  const query={id:'q',node_id:'n',path:'/fixture/complete',status:'READY',inspection:fixtures.complete,error:null};
  assert.equal(h.inspectionMatches(query,'n','/fixture//./complete/','q'),true);
  for(const [node,path,id]of[['other','/fixture/complete','q'],['n','/other','q'],['n','/fixture/complete','later']])assert.equal(h.inspectionMatches(query,node,path,id),false);
  assert.equal(h.inspectionMatches({...query,inspection:{...fixtures.complete,path:'/stale'}},'n','/fixture/complete'),false);
 });
 await test('offline UI keeps manual retry visible and explains the Agent requirement without assuming a deployment',()=>{
  const html=renderToStaticMarkup(React.createElement(DatasetInspection,{node:{id:'n',status:'OFFLINE'},path:'/fixture/complete',value:emptyDatasetProfile(),draft:{apply(){},resetScope(){}}}));
  assert.match(html,/读取数据集字段，推断并填入默认值/);assert.match(html,/机器当前离线/);assert.match(html,/可继续手动填写，在线后重试/);assert.match(html,/<button[^>]*disabled/);assert.doesNotMatch(html,/本地元数据快照|type="submit"/);
 });
}finally{await server.close();}
