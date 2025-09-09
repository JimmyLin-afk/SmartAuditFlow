import { Delegation } from '@ethereum-attestation-service/eas-sdk'
import { ecsign } from 'ethereumjs-util'
import { BigNumber, Signer } from 'ethers'

const HARDHAT_CHAIN_ID = 31337

export class EIP712Utils {
  delegation: Delegation
  constructor(contractAddress: string) {
    this.delegation = new Delegation({
      address: contractAddress,
      version: '0.8',
      chainId: HARDHAT_CHAIN_ID,
    })
  }

  /*
  How does the domain separator get put in here?
  How can we prove this even works at all ?s
  */

  // eslint-disable-next-line @typescript-eslint/explicit-function-return-type
  async getAttestationRequest(
    recipientAddress: string,
    schema: string,
    expirationTime: BigNumber,
    refUUID: string,
    data: string,
    nonce: BigNumber,
    privateKey: string
  ) {
    return await this.delegation.getAttestationRequest(
      {
        recipient: recipientAddress,
        schema,
        expirationTime,
        refUUID,
        data: Buffer.from(data.slice(2), 'hex'),
        nonce,
      },
      async (message) => {
        const { v, r, s } = ecsign(
          message,
          Buffer.from(privateKey.slice(2), 'hex')
        )
        return { v, r, s }
      }
    )
  }

  // eslint-disable-next-line @typescript-eslint/explicit-function-return-type
  async getRevocationRequest(
    uuid: string,
    nonce: BigNumber,
    signer: Signer,
    privateKey: string
  ) {
    return await this.delegation.getRevocationRequest(
      {
        uuid,
        nonce,
      },
      async (message) => {
        const { v, r, s } = ecsign(
          message,
          Buffer.from(privateKey.slice(2), 'hex')
        )
        return { v, r, s }
      }
    )
  }

  /*



  async recoverAttestationRequest(
    recipient: Wallet,
    schema: string,
    expirationTime: BigNumber,
    refUUID: string,
    data: string,
    nonce:BigNumber,
    v: number,
    r: Buffer,
    s: Buffer

  ){

    let typedData = {
      types: {
        EIP712Domain: [
            { name: "contractName", type: "string" },
            { name: "version", type: "string" },
            { name: "chainId", type: "uint256" },
            { name: "verifyingContract", type: "address" }
        ]
        },
        primaryType: customConfig.primaryType,
        domain: {
            contractName: customConfig.contractName,
            version: customConfig.version,
            chainId: _chainId,  
            verifyingContract: web3utils.toChecksumAddress(_contractAddress)
        },
        message: dataValues  
    }


    var typedDataHash = keccak256(
      Buffer.concat([
          Buffer.from('1901', 'hex'),
          EIP712Utils.structHash('EIP712Domain', typedData.domain, typedData.types),
          EIP712Utils.structHash(typedData.primaryType, typedData.message, typedData.types),
      ])
    )

    let pubKey = ecrecover(toBuffer(typedDataHash), v,r,s )
    
    const addrBuf = pubToAddress(pubKey);
    const outputAddress    = bufferToHex(addrBuf);
       
    return outputAddress
  }


   // Recursively finds all the dependencies of a type
   static  dependencies(primaryType:string, types:any, found = []) {
    if (found.includes(primaryType)) {
        return found;
    }
    if (types[primaryType] === undefined) {
        return found;
    }
    found.push(primaryType);
    for (let field of types[primaryType]) {
        for (let dep of EIP712Utils.dependencies(field.type, types, found)) {
            if (!found.includes(dep)) {
                found.push(dep);
            }
        }
    }
    return found;
}

  static  encodeType(primaryType:string, types:any) {
    // Get dependencies primary first, then alphabetical
    let deps = EIP712Utils.dependencies(primaryType, types);
    deps = deps.filter(t => t != primaryType);
    deps = [primaryType].concat(deps.sort());

    // Format as a string with fields
    let result = '';
    for (let type of deps) {
        result += `${type}(${types[type].map(({ name, type }) => `${type} ${name}`).join(',')})`;
    }

    return result;
}

static typeHash(primaryType:string, types:any) {
    return keccak256( Buffer.from(EIP712Utils.encodeType(primaryType, types)));
}


  static encodeData(primaryType:string, data:any, types:any) {
    let encTypes = [];
    let encValues = [];

    // Add typehash
    encTypes.push('bytes32');
    encValues.push(EIP712Utils.typeHash(primaryType, types));

    //console.log('typehash 1  ', Buffer.from( EIP712Helper.typeHash(primaryType, types) ).toString('hex'))

    // Add field contents
    for (let field of types[primaryType]) {
        let value = data[field.name];
        if (field.type == 'string' || field.type == 'bytes') {
            encTypes.push('bytes32');
            value = keccak256(Buffer.from(value));

            //console.log('typehash 2  ', value)

            encValues.push(value);
        } else if (types[field.type] !== undefined) {
            encTypes.push('bytes32');
            value = keccak256(Buffer.from(EIP712Utils.encodeData(field.type, value, types)));
            encValues.push(value);
        } else if (field.type.lastIndexOf(']') === field.type.length - 1) {
            throw 'TODO: Arrays currently unimplemented in encodeData';
        } else {
            encTypes.push(field.type);
            encValues.push(value);
        }
    }
  //  console.log('encValues',encValues)
    return abi.rawEncode(encTypes, encValues);
}


  static structHash(primaryType:string, data:any, types:any) {
      return keccak256(Buffer.from(EIP712Utils.encodeData(primaryType, data, types)));
  }



  */
}
