const fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
global.window=global;
vm.runInThisContext(fs.readFileSync(path.join(__dirname,'dist/model.js'),'utf8'));
require('./dist/model-runtime.js');
if(!process.argv[2])throw new Error('Pass the local parity-fixture.json path');
const fixture=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
let maxError=0;
fixture.x.forEach((x,i)=>{const actual=BikeModelRuntime.predict(BIKE_MODEL,x),expected=fixture.expected[i];maxError=Math.max(maxError,Math.abs(actual-expected));if(Math.abs(actual-expected)>1e-7)throw new Error(`Prediction mismatch at ${i}: ${actual} vs ${expected}`);});
console.log(JSON.stringify({scenarios:fixture.x.length,maxAbsoluteError:maxError,selected:BIKE_MODEL.name}));
